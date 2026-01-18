from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _patch_rediscluster():
    try:
        import rediscluster
    except ImportError:
        return
    patch_redis_client(rediscluster.RedisCluster, is_cluster=True, set_db_data_fn=_set_db_data)
    version = getattr(rediscluster, 'VERSION', rediscluster.__version__)
    if (0, 2, 0) < version < (2, 0, 0):
        pipeline_cls = rediscluster.pipeline.StrictClusterPipeline
        patch_redis_client(rediscluster.StrictRedisCluster, is_cluster=True, set_db_data_fn=_set_db_data)
    else:
        pipeline_cls = rediscluster.pipeline.ClusterPipeline
    patch_redis_pipeline(pipeline_cls, True, _parse_rediscluster_command, set_db_data_fn=_set_db_data)