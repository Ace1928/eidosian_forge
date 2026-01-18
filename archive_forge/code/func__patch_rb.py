from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _patch_rb():
    try:
        import rb.clients
    except ImportError:
        pass
    else:
        patch_redis_client(rb.clients.FanoutClient, is_cluster=False, set_db_data_fn=_set_db_data)
        patch_redis_client(rb.clients.MappingClient, is_cluster=False, set_db_data_fn=_set_db_data)
        patch_redis_client(rb.clients.RoutingClient, is_cluster=False, set_db_data_fn=_set_db_data)