import asyncio
import collections
import random
import socket
import warnings
from typing import (
from aiokeydb.v1.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from aiokeydb.v1.asyncio.core import ResponseCallbackT
from aiokeydb.v1.asyncio.connection import (
from aiokeydb.v1.asyncio.parser import CommandsParser
from aiokeydb.v1.core import EMPTY_RESPONSE, NEVER_DECODE, AbstractKeyDB
from aiokeydb.v1.cluster import (
from aiokeydb.v1.commands import READ_COMMANDS, AsyncKeyDBClusterCommands
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.typing import AnyKeyT, EncodableT, KeyT
from aiokeydb.v1.utils import dict_merge, safe_str, str_if_bytes
class AsyncClusterPipeline(AbstractKeyDB, AbstractKeyDBCluster, AsyncKeyDBClusterCommands):
    """
    Create a new AsyncClusterPipeline object.

    Usage::

        result = await (
            rc.pipeline()
            .set("A", 1)
            .get("A")
            .hset("K", "F", "V")
            .hgetall("K")
            .mset_nonatomic({"A": 2, "B": 3})
            .get("A")
            .get("B")
            .delete("A", "B", "K")
            .execute()
        )
        # result = [True, "1", 1, {"F": "V"}, True, True, "2", "3", 1, 1, 1]

    Note: For commands `DELETE`, `EXISTS`, `TOUCH`, `UNLINK`, `mset_nonatomic`, which
    are split across multiple nodes, you'll get multiple results for them in the array.

    Retryable errors:
        - :class:`~.ClusterDownError`
        - :class:`~.ConnectionError`
        - :class:`~.TimeoutError`

    Redirection errors:
        - :class:`~.TryAgainError`
        - :class:`~.MovedError`
        - :class:`~.AskError`

    :param client:
        | Existing :class:`~.AsyncKeyDBCluster` client
    """
    __slots__ = ('_command_stack', '_client')

    def __init__(self, client: AsyncKeyDBCluster) -> None:
        self._client = client
        self._command_stack: List['PipelineCommand'] = []

    async def initialize(self) -> 'AsyncClusterPipeline':
        if self._client._initialize:
            await self._client.initialize()
        self._command_stack = []
        return self

    async def __aenter__(self) -> 'AsyncClusterPipeline':
        return await self.initialize()

    async def __aexit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
        self._command_stack = []

    def __await__(self) -> Generator[Any, None, 'AsyncClusterPipeline']:
        return self.initialize().__await__()

    def __enter__(self) -> 'AsyncClusterPipeline':
        self._command_stack = []
        return self

    def __exit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
        self._command_stack = []

    def __bool__(self) -> bool:
        return bool(self._command_stack)

    def __len__(self) -> int:
        return len(self._command_stack)

    def execute_command(self, *args: Union[KeyT, EncodableT], **kwargs: Any) -> 'AsyncClusterPipeline':
        """
        Append a raw command to the pipeline.

        :param args:
            | Raw command args
        :param kwargs:

            - target_nodes: :attr:`NODE_FLAGS` or :class:`~.AsyncClusterNode`
              or List[:class:`~.AsyncClusterNode`] or Dict[Any, :class:`~.AsyncClusterNode`]
            - Rest of the kwargs are passed to the KeyDB connection
        """
        self._command_stack.append(PipelineCommand(len(self._command_stack), *args, **kwargs))
        return self

    async def execute(self, raise_on_error: bool=True, allow_redirections: bool=True) -> List[Any]:
        """
        Execute the pipeline.

        It will retry the commands as specified by :attr:`cluster_error_retry_attempts`
        & then raise an exception.

        :param raise_on_error:
            | Raise the first error if there are any errors
        :param allow_redirections:
            | Whether to retry each failed command individually in case of redirection
              errors

        :raises KeyDBClusterException: if target_nodes is not provided & the command
            can't be mapped to a slot
        """
        if not self._command_stack:
            return []
        try:
            for _ in range(self._client.cluster_error_retry_attempts):
                if self._client._initialize:
                    await self._client.initialize()
                try:
                    return await self._execute(self._client, self._command_stack, raise_on_error=raise_on_error, allow_redirections=allow_redirections)
                except BaseException as e:
                    if type(e) in self.__class__.ERRORS_ALLOW_RETRY:
                        exception = e
                        await self._client.close()
                        await asyncio.sleep(0.25)
                    else:
                        raise
            raise exception
        finally:
            self._command_stack = []

    async def _execute(self, client: 'AsyncKeyDBCluster', stack: List['PipelineCommand'], raise_on_error: bool=True, allow_redirections: bool=True) -> List[Any]:
        todo = [cmd for cmd in stack if not cmd.result or isinstance(cmd.result, Exception)]
        nodes = {}
        for cmd in todo:
            passed_targets = cmd.kwargs.pop('target_nodes', None)
            if passed_targets and (not client._is_node_flag(passed_targets)):
                target_nodes = client._parse_target_nodes(passed_targets)
            else:
                target_nodes = await client._determine_nodes(*cmd.args, node_flag=passed_targets)
                if not target_nodes:
                    raise KeyDBClusterException(f'No targets were found to execute {cmd.args} command on')
            if len(target_nodes) > 1:
                raise KeyDBClusterException(f'Too many targets for command {cmd.args}')
            node = target_nodes[0]
            if node.name not in nodes:
                nodes[node.name] = (node, [])
            nodes[node.name][1].append(cmd)
        errors = await asyncio.gather(*(asyncio.ensure_future(node[0].execute_pipeline(node[1])) for node in nodes.values()))
        if any(errors):
            if allow_redirections:
                for cmd in todo:
                    if isinstance(cmd.result, (TryAgainError, MovedError, AskError)):
                        try:
                            cmd.result = await client.execute_command(*cmd.args, **cmd.kwargs)
                        except Exception as e:
                            cmd.result = e
            if raise_on_error:
                for cmd in todo:
                    result = cmd.result
                    if isinstance(result, Exception):
                        command = ' '.join(map(safe_str, cmd.args))
                        msg = f'Command # {cmd.position + 1} ({command}) of pipeline caused error: {result.args}'
                        result.args = (msg,) + result.args[1:]
                        raise result
        return [cmd.result for cmd in stack]

    def _split_command_across_slots(self, command: str, *keys: KeyT) -> 'AsyncClusterPipeline':
        for slot_keys in self._client._partition_keys_by_slot(keys).values():
            self.execute_command(command, *slot_keys)
        return self

    def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> 'AsyncClusterPipeline':
        encoder = self._client.encoder
        slots_pairs = {}
        for pair in mapping.items():
            slot = key_slot(encoder.encode(pair[0]))
            slots_pairs.setdefault(slot, []).extend(pair)
        for pairs in slots_pairs.values():
            self.execute_command('MSET', *pairs)
        return self