import sys
import time
import anyio
import typing
import logging
import asyncio
import functools
import contextlib
from pydantic import BaseModel
from pydantic.types import ByteSize
from aiokeydb.v2.typing import Number, KeyT, ExpiryT, AbsExpiryT, PatternT
from aiokeydb.v2.lock import Lock, AsyncLock
from aiokeydb.v2.core import KeyDB, PubSub, Pipeline, PipelineT, PubSubT
from aiokeydb.v2.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline, AsyncPipelineT, AsyncPubSubT
from aiokeydb.v2.connection import Encoder, ConnectionPool, AsyncConnectionPool
from aiokeydb.v2.exceptions import (
from aiokeydb.v2.types import KeyDBUri, ENOVAL
from aiokeydb.v2.configs import KeyDBSettings, settings as default_settings
from aiokeydb.v2.utils import full_name, args_to_key
from aiokeydb.v2.utils.helpers import create_retryable_client
from aiokeydb.v2.serializers import BaseSerializer
from inspect import iscoroutinefunction
def _cli(self, args: typing.Union[str, typing.List[str]], shell: bool=True, raise_error: bool=True, entrypoint: str='keydb-cli', **kwargs) -> str:
    """
        Runs a CLI command on the server
        """
    base_args = self.uri.connection_args.copy()
    if not isinstance(args, list):
        args = [args]
    base_args.extend(args)
    command = ' '.join(base_args)
    if '-n' not in command:
        command = f'{command} -n {self.uri.db_id}'
    if entrypoint not in command:
        command = f'{entrypoint} {command}'
    import subprocess
    try:
        out = subprocess.check_output(command, shell=shell, **kwargs)
        if isinstance(out, bytes):
            out = out.decode('utf8')
        return out.strip()
    except Exception as e:
        if not raise_error:
            return ''
        raise e