from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _patch_redis(StrictRedis, client):
    patch_redis_client(StrictRedis, is_cluster=False, set_db_data_fn=_set_db_data)
    patch_redis_pipeline(client.Pipeline, False, _get_redis_command_args, _set_db_data)
    try:
        strict_pipeline = client.StrictPipeline
    except AttributeError:
        pass
    else:
        patch_redis_pipeline(strict_pipeline, False, _get_redis_command_args, _set_db_data)
    try:
        import redis.asyncio
    except ImportError:
        pass
    else:
        from sentry_sdk.integrations.redis.asyncio import patch_redis_async_client, patch_redis_async_pipeline
        patch_redis_async_client(redis.asyncio.client.StrictRedis, is_cluster=False, set_db_data_fn=_set_db_data)
        patch_redis_async_pipeline(redis.asyncio.client.Pipeline, False, _get_redis_command_args, set_db_data_fn=_set_db_data)