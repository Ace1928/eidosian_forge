from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _patch_redis_cluster():
    """Patches the cluster module on redis SDK (as opposed to rediscluster library)"""
    try:
        from redis import RedisCluster, cluster
    except ImportError:
        pass
    else:
        patch_redis_client(RedisCluster, True, _set_cluster_db_data)
        patch_redis_pipeline(cluster.ClusterPipeline, True, _parse_rediscluster_command, _set_cluster_db_data)
    try:
        from redis.asyncio import cluster as async_cluster
    except ImportError:
        pass
    else:
        from sentry_sdk.integrations.redis.asyncio import patch_redis_async_client, patch_redis_async_pipeline
        patch_redis_async_client(async_cluster.RedisCluster, is_cluster=True, set_db_data_fn=_set_async_cluster_db_data)
        patch_redis_async_pipeline(async_cluster.ClusterPipeline, True, _parse_rediscluster_command, set_db_data_fn=_set_async_cluster_pipeline_db_data)