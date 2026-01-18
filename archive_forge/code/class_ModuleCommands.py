import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class ModuleCommands(CommandsProtocol):
    """
    Redis Module commands.
    see: https://redis.io/topics/modules-intro
    """

    def module_load(self, path, *args) -> ResponseT:
        """
        Loads the module from ``path``.
        Passes all ``*args`` to the module, during loading.
        Raises ``ModuleError`` if a module is not found at ``path``.

        For more information see https://redis.io/commands/module-load
        """
        return self.execute_command('MODULE LOAD', path, *args)

    def module_loadex(self, path: str, options: Optional[List[str]]=None, args: Optional[List[str]]=None) -> ResponseT:
        """
        Loads a module from a dynamic library at runtime with configuration directives.

        For more information see https://redis.io/commands/module-loadex
        """
        pieces = []
        if options is not None:
            pieces.append('CONFIG')
            pieces.extend(options)
        if args is not None:
            pieces.append('ARGS')
            pieces.extend(args)
        return self.execute_command('MODULE LOADEX', path, *pieces)

    def module_unload(self, name) -> ResponseT:
        """
        Unloads the module ``name``.
        Raises ``ModuleError`` if ``name`` is not in loaded modules.

        For more information see https://redis.io/commands/module-unload
        """
        return self.execute_command('MODULE UNLOAD', name)

    def module_list(self) -> ResponseT:
        """
        Returns a list of dictionaries containing the name and version of
        all loaded modules.

        For more information see https://redis.io/commands/module-list
        """
        return self.execute_command('MODULE LIST')

    def command_info(self) -> None:
        raise NotImplementedError('COMMAND INFO is intentionally not implemented in the client.')

    def command_count(self) -> ResponseT:
        return self.execute_command('COMMAND COUNT')

    def command_getkeys(self, *args) -> ResponseT:
        return self.execute_command('COMMAND GETKEYS', *args)

    def command(self) -> ResponseT:
        return self.execute_command('COMMAND')