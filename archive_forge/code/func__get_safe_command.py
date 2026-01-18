from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _get_safe_command(name, args):
    command_parts = [name]
    for i, arg in enumerate(args):
        if i > _MAX_NUM_ARGS:
            break
        name_low = name.lower()
        if name_low in _COMMANDS_INCLUDING_SENSITIVE_DATA:
            command_parts.append(SENSITIVE_DATA_SUBSTITUTE)
            continue
        arg_is_the_key = i == 0
        if arg_is_the_key:
            command_parts.append(repr(arg))
        elif _should_send_default_pii():
            command_parts.append(repr(arg))
        else:
            command_parts.append(SENSITIVE_DATA_SUBSTITUTE)
    command = ' '.join(command_parts)
    return command