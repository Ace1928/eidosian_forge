import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
def _init_process_groups(self):
    dim_group_infos: List[Tuple[str, List[int]]] = []
    if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
        dim_group_infos.append((_get_group_tag(_get_default_group()), list(range(get_world_size()))))
    else:
        for dim in range(self.mesh.ndim):
            pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
            for dim_mesh in pg_ranks_by_dim:
                subgroup_ranks = dim_mesh.tolist()
                dim_group = new_group(ranks=subgroup_ranks)
                if self.get_rank() in subgroup_ranks:
                    if len(dim_group_infos) > dim:
                        raise RuntimeError(f'Each device mesh dimension should get only one process group, but got {self.get_rank} in {subgroup_ranks}!')
                    dim_group_infos.append((_get_group_tag(dim_group), subgroup_ranks))
    self._dim_group_infos = dim_group_infos