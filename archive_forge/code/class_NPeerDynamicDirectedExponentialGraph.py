from abc import ABC, abstractmethod
from math import log as mlog
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
class NPeerDynamicDirectedExponentialGraph(GraphManager):

    def _make_graph(self) -> None:
        for rank in range(self.world_size):
            for i in range(0, int(mlog(self.world_size - 1, self._peers_per_itr + 1)) + 1):
                for j in range(1, self._peers_per_itr + 1):
                    distance_to_neighbor = j * (self._peers_per_itr + 1) ** i
                    f_peer = self._rotate_forward(rank, distance_to_neighbor)
                    self._add_peers(rank, [f_peer])

    def is_regular_graph(self) -> bool:
        return True

    def is_bipartite_graph(self) -> bool:
        return False

    def is_passive(self, rank: Optional[int]=None) -> bool:
        return False

    def is_dynamic_graph(self) -> bool:
        return True