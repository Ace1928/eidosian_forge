from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _set_async_cluster_db_data(span, async_redis_cluster_instance):
    default_node = async_redis_cluster_instance.get_default_node()
    if default_node is not None and default_node.connection_kwargs is not None:
        _set_db_data_on_span(span, default_node.connection_kwargs)