import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
class BatchingNodeProvider(NodeProvider):
    """Abstract subclass of NodeProvider meant for use with external cluster managers.

    Batches reads of cluster state into a single method, get_node_data, called at the
    start of an autoscaling update.

    Batches modifications to cluster state into a single method, submit_scale_request,
    called at the end of an autoscaling update.

    Implementing a concrete subclass of BatchingNodeProvider only requires overriding
    get_node_data() and submit_scale_request().

    See the method docstrings for more information.

    Note that an autoscaling update may be conditionally
    cancelled using the optional method safe_to_scale()
    of the root NodeProvider.
    """

    def __init__(self, provider_config: Dict[str, Any], cluster_name: str, _allow_multiple: bool=False) -> None:
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.node_data_dict: Dict[NodeID, NodeData] = {}
        global provider_exists
        if not _allow_multiple:
            assert not provider_exists, 'Only one BatchingNodeProvider allowed per process.'
        assert provider_config.get(DISABLE_NODE_UPDATERS_KEY, False) is True, f'To use BatchingNodeProvider, must set `{DISABLE_NODE_UPDATERS_KEY}:True`.'
        assert provider_config.get(DISABLE_LAUNCH_CONFIG_CHECK_KEY, False) is True, f'To use BatchingNodeProvider, must set `{DISABLE_LAUNCH_CONFIG_CHECK_KEY}:True`.'
        assert provider_config.get(FOREGROUND_NODE_LAUNCH_KEY, False) is True, f'To use BatchingNodeProvider, must set `{FOREGROUND_NODE_LAUNCH_KEY}:True`.'
        self.scale_change_needed = False
        self.scale_request = ScaleRequest()

    def get_node_data(self) -> Dict[NodeID, NodeData]:
        """Queries cluster manager for node info. Returns a mapping from node id to
        NodeData.

        Each NodeData value must adhere to the semantics of the NodeData docstring.
        (Note in particular the requirements for NodeData.status.)

        Consistency requirement:
        If a node id was present in ScaleRequest.workers_to_delete of a previously
        submitted scale request, it should no longer be present as a key in
        get_node_data.
        (Node termination must be registered immediately when submit_scale_request
        returns.)
        """
        raise NotImplementedError

    def submit_scale_request(self, scale_request: ScaleRequest) -> None:
        """Tells the cluster manager which nodes to delete and how many nodes of
        each node type to maintain.

        Consistency requirement:
        If a node id was present in ScaleRequest.workers_to_delete of a previously
        submitted scale request, it should no longer be present as key in get_node_data.
        (Node termination must be registered immediately when submit_scale_request
        returns.)
        """
        raise NotImplementedError

    def post_process(self) -> None:
        """Submit a scale request if it is necessary to do so."""
        if self.scale_change_needed:
            self.submit_scale_request(self.scale_request)
        self.scale_change_needed = False

    def non_terminated_nodes(self, tag_filters: Dict[str, str]) -> List[str]:
        self.scale_change_needed = False
        self.node_data_dict = self.get_node_data()
        self.scale_request = ScaleRequest(desired_num_workers=self.cur_num_workers(), workers_to_delete=set())
        all_nodes = list(self.node_data_dict.keys())
        filtered_nodes = [node for node in all_nodes if tag_filters.items() <= self.node_tags(node).items()]
        return filtered_nodes

    def cur_num_workers(self):
        """Returns dict mapping node type to the number of nodes of that type."""
        return self._cur_num_workers(self.node_data_dict)

    def _cur_num_workers(self, node_data_dict: Dict[str, Any]):
        num_workers_dict = defaultdict(int)
        for node_data in node_data_dict.values():
            if node_data.kind == NODE_KIND_HEAD:
                continue
            num_workers_dict[node_data.type] += 1
        return num_workers_dict

    def node_tags(self, node_id: str) -> Dict[str, str]:
        node_data = self.node_data_dict[node_id]
        return {TAG_RAY_NODE_KIND: node_data.kind, TAG_RAY_NODE_STATUS: node_data.status, TAG_RAY_USER_NODE_TYPE: node_data.type}

    def internal_ip(self, node_id: str) -> str:
        return self.node_data_dict[node_id].ip

    def create_node(self, node_config: Dict[str, Any], tags: Dict[str, str], count: int) -> Optional[Dict[str, Any]]:
        node_type = tags[TAG_RAY_USER_NODE_TYPE]
        self.scale_request.desired_num_workers[node_type] += count
        self.scale_change_needed = True

    def terminate_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id in self.scale_request.workers_to_delete:
            logger.warning(f'Autoscaler tried to terminate node {node_id} twice in the same update. Skipping termination request.')
            return
        if node_id not in self.node_data_dict:
            logger.warning(f'Autoscaler tried to terminate unkown node {node_id}. Skipping termination request.')
            return
        node_type = self.node_data_dict[node_id].type
        if self.scale_request.desired_num_workers[node_type] <= 0:
            raise AssertionError(f'NodeProvider attempted to request less than 0 workers of type {node_type}. Skipping termination request.')
        self.scale_request.desired_num_workers[node_type] -= 1
        self.scale_request.workers_to_delete.add(node_id)
        self.scale_change_needed = True