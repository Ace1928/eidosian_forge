import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def _gen_placement_group_info(self, placement_group_info):
    from ray.core.generated.common_pb2 import PlacementStrategy

    def get_state(state):
        if state == gcs_pb2.PlacementGroupTableData.PENDING:
            return 'PENDING'
        elif state == gcs_pb2.PlacementGroupTableData.CREATED:
            return 'CREATED'
        elif state == gcs_pb2.PlacementGroupTableData.RESCHEDULING:
            return 'RESCHEDULING'
        else:
            return 'REMOVED'

    def get_strategy(strategy):
        if strategy == PlacementStrategy.PACK:
            return 'PACK'
        elif strategy == PlacementStrategy.STRICT_PACK:
            return 'STRICT_PACK'
        elif strategy == PlacementStrategy.STRICT_SPREAD:
            return 'STRICT_SPREAD'
        elif strategy == PlacementStrategy.SPREAD:
            return 'SPREAD'
        else:
            raise ValueError(f'Invalid strategy returned: {PlacementStrategy}')
    stats = placement_group_info.stats
    assert placement_group_info is not None
    return {'placement_group_id': binary_to_hex(placement_group_info.placement_group_id), 'name': placement_group_info.name, 'bundles': {bundle.bundle_id.bundle_index: message_to_dict(bundle)['unitResources'] for bundle in placement_group_info.bundles}, 'bundles_to_node_id': {bundle.bundle_id.bundle_index: binary_to_hex(bundle.node_id) for bundle in placement_group_info.bundles}, 'strategy': get_strategy(placement_group_info.strategy), 'state': get_state(placement_group_info.state), 'stats': {'end_to_end_creation_latency_ms': stats.end_to_end_creation_latency_us / 1000.0, 'scheduling_latency_ms': stats.scheduling_latency_us / 1000.0, 'scheduling_attempt': stats.scheduling_attempt, 'highest_retry_delay_ms': stats.highest_retry_delay_ms, 'scheduling_state': gcs_pb2.PlacementGroupStats.SchedulingState.DESCRIPTOR.values_by_number[stats.scheduling_state].name}}