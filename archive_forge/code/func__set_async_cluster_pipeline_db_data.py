from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _set_async_cluster_pipeline_db_data(span, async_redis_cluster_pipeline_instance):
    with capture_internal_exceptions():
        _set_async_cluster_db_data(span, async_redis_cluster_pipeline_instance._client)