from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _set_pipeline_data(span, is_cluster, get_command_args_fn, is_transaction, command_stack):
    span.set_tag('redis.is_cluster', is_cluster)
    span.set_tag('redis.transaction', is_transaction)
    commands = []
    for i, arg in enumerate(command_stack):
        if i >= _MAX_NUM_COMMANDS:
            break
        command = get_command_args_fn(arg)
        commands.append(_get_safe_command(command[0], command[1:]))
    span.set_data('redis.commands', {'count': len(command_stack), 'first_ten': commands})