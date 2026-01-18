import copy
import re
import threading
import time
import warnings
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type, Union
from redis._parsers.encoders import Encoder
from redis._parsers.helpers import (
from redis.commands import (
from redis.connection import (
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def _execute_transaction(self, connection, commands, raise_on_error) -> List:
    cmds = chain([(('MULTI',), {})], commands, [(('EXEC',), {})])
    all_cmds = connection.pack_commands([args for args, options in cmds if EMPTY_RESPONSE not in options])
    connection.send_packed_command(all_cmds)
    errors = []
    try:
        self.parse_response(connection, '_')
    except ResponseError as e:
        errors.append((0, e))
    for i, command in enumerate(commands):
        if EMPTY_RESPONSE in command[1]:
            errors.append((i, command[1][EMPTY_RESPONSE]))
        else:
            try:
                self.parse_response(connection, '_')
            except ResponseError as e:
                self.annotate_exception(e, i + 1, command[0])
                errors.append((i, e))
    try:
        response = self.parse_response(connection, '_')
    except ExecAbortError:
        if errors:
            raise errors[0][1]
        raise
    self.watching = False
    if response is None:
        raise WatchError('Watched variable changed.')
    for i, e in errors:
        response.insert(i, e)
    if len(response) != len(commands):
        self.connection.disconnect()
        raise ResponseError('Wrong number of response items from pipeline execution')
    if raise_on_error:
        self.raise_first_error(commands, response)
    data = []
    for r, cmd in zip(response, commands):
        if not isinstance(r, Exception):
            args, options = cmd
            command_name = args[0]
            if command_name in self.response_callbacks:
                r = self.response_callbacks[command_name](r, **options)
        data.append(r)
    return data