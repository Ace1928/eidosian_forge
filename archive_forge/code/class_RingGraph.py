from abc import ABC, abstractmethod
from math import log as mlog
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
class RingGraph(GraphManager):

    def _make_graph(self) -> None:
        for rank in range(self.world_size):
            f_peer = self._rotate_forward(rank, 1)
            b_peer = self._rotate_backward(rank, 1)
            self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self) -> bool:
        return True

    def is_bipartite_graph(self) -> bool:
        return False

    def is_passive(self, rank: Optional[int]=None) -> bool:
        return False

    def is_dynamic_graph(self) -> bool:
        return False