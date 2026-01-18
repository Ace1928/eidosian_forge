import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
class _MeshEnv:

    def __init__(self) -> None:
        self.mesh_stack: List[DeviceMesh] = []
        self.child_to_parent_mapping: Dict[DeviceMesh, DeviceMesh] = {}

    def get_current_mesh(self) -> 'DeviceMesh':
        if len(self.mesh_stack) == 0:
            raise RuntimeError('No device mesh is currently active!')
        return self.mesh_stack[-1]

    def create_child_mesh(self, device_mesh: 'DeviceMesh', mesh_dim: int, mesh_dim_name: str) -> 'DeviceMesh':
        cur_rank = device_mesh.get_rank()
        pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(-1, device_mesh.mesh.size(mesh_dim))
        for mesh_1d in pg_ranks_by_dim:
            sub_mesh = DeviceMesh(device_mesh.device_type, mesh_1d, mesh_dim_names=(mesh_dim_name,), _init_process_groups=False)
            if cur_rank in mesh_1d:
                res_sub_mesh = sub_mesh
        res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[mesh_dim]]
        self.child_to_parent_mapping[res_sub_mesh] = device_mesh
        return res_sub_mesh

    def get_parent_mesh(self, device_mesh: 'DeviceMesh') -> Optional['DeviceMesh']:
        return self.child_to_parent_mapping.get(device_mesh, None)

    def get_parent_mesh_dim(self, device_mesh: 'DeviceMesh') -> Optional[int]:
        """
            Return the index of the mesh dim in the parent mesh.
            The device_mesh passed in needs to be sliced out from a parent mesh.
            """
        parent_mesh = self.get_parent_mesh(device_mesh)
        child_mesh_dim_names = device_mesh.mesh_dim_names
        if parent_mesh and child_mesh_dim_names:
            assert len(child_mesh_dim_names) == 1, 'The child mesh can only be a 1D mesh.'
            child_mesh_dim_name = child_mesh_dim_names[0]
            if parent_mesh.mesh_dim_names:
                return parent_mesh._get_mesh_dim_by_name(child_mesh_dim_name)
        return None

    @staticmethod
    def num_devices_per_host(device_type: str) -> int:
        return _get_device_handle(device_type).device_count()

    @staticmethod
    def num_hosts(device_type: str) -> int:
        return get_world_size() // _MeshEnv.num_devices_per_host(device_type)