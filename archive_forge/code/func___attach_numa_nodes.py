from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __attach_numa_nodes(self, entity):
    numa_nodes_service = self._service.service(entity.id).numa_nodes_service()
    existed_numa_nodes = numa_nodes_service.list()
    if len(self.param('numa_nodes')) > 0:
        for current_numa_node in sorted(existed_numa_nodes, reverse=True, key=lambda x: x.index):
            numa_nodes_service.node_service(current_numa_node.id).remove()
    for numa_node in self.param('numa_nodes'):
        if numa_node is None or numa_node.get('index') is None or numa_node.get('cores') is None or (numa_node.get('memory') is None):
            continue
        numa_nodes_service.add(otypes.VirtualNumaNode(index=numa_node.get('index'), memory=numa_node.get('memory'), cpu=otypes.Cpu(cores=[otypes.Core(index=core) for core in numa_node.get('cores')]), numa_node_pins=[otypes.NumaNodePin(index=pin) for pin in numa_node.get('numa_node_pins')] if numa_node.get('numa_node_pins') is not None else None))
    return self.__get_numa_serialized(numa_nodes_service.list()) != self.__get_numa_serialized(existed_numa_nodes)