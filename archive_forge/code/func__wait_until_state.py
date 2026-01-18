import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _wait_until_state(self, nodes, state, wait_period=3, timeout=600):
    """
        Block until the provided nodes are in the desired state.
        :param nodes: List of nodes to wait for
        :type nodes: ``list`` of :class:`.Node`
        :param state: desired state
        :type state: ``NodeState``
        :param wait_period: How many seconds to wait between each loop
                            iteration. (default is 3)
        :type wait_period: ``int``
        :param timeout: How many seconds to wait before giving up.
                        (default is 600)
        :type timeout: ``int``
        :return: if the nodes are in the desired state.
        :rtype: ``bool``
        """
    start = time.time()
    end = start + timeout
    node_ids = [node.id for node in nodes]
    while time.time() < end:
        matched_nodes = self.list_nodes(ex_node_ids=node_ids)
        if len(matched_nodes) > len(node_ids):
            found_ids = [node.id for node in matched_nodes]
            msg = 'found multiple nodes with same ids, desired ids: %(ids)s, found ids: %(found_ids)s' % {'ids': node_ids, 'found_ids': found_ids}
            raise LibcloudError(value=msg, driver=self)
        desired_nodes = [node for node in matched_nodes if node.state == state]
        if len(desired_nodes) == len(node_ids):
            return True
        else:
            time.sleep(wait_period)
            continue
    raise LibcloudError(value='Timed out after %s seconds' % timeout, driver=self)