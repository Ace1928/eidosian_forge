from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
class Gossiper(object):
    """Generic gossip averaging object for multi-peer communication

    Args:
        msg (torch.Tensor): message used to initialize recv buffer
        graph (GraphManager): Subclass of GraphManager
        device: (torch.Device) device on which to initialize recv buffer
        mixing (MixingManager): Subclass of MixingManager
        logger (logging.Logger): Module used to log results
        rank (int): Rank of the current process
        world_size (int): World size of the current process
    """

    def __init__(self, msg: torch.Tensor, graph: GraphManager, device: Optional[torch.device]=None, mixing: MixingManager=None, logger: logging.Logger=None, rank: Optional[int]=None, world_size: Optional[int]=None) -> None:
        """
        Initialize generic averaging class designed for multi-peer comms
        """
        self.logger = logger
        if rank is None or world_size is None:
            assert dist.is_initialized()
            assert dist.get_backend() != dist_backend.GLOO
            assert dist.get_backend() != dist_backend.NCCL
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.rank = rank
        self.world_size = world_size
        assert isinstance(graph, GraphManager)
        self._graph_manager = graph
        self.peers_per_itr_device = torch.tensor([self._graph_manager.peers_per_itr], device=device, dtype=msg.dtype)
        self.passive = self._graph_manager.is_passive()
        self.refresh_peers_(rotate=False)
        if mixing is None:
            mixing = UniformMixing(self._graph_manager, device)
        assert isinstance(mixing, MixingManager)
        self._mixing_manager = mixing
        self.refresh_mixing_weights_()
        self.regular = self._mixing_manager.is_regular()
        self.device = device if device is not None else msg.device
        self.out_msg_buffer: List[Tuple[dist.Work, torch.Tensor]] = []
        self.in_msg_buffer = msg.clone().detach_().to(self.device)
        self._ps_weight: torch.Tensor = torch.ones(1, dtype=msg.dtype).detach_().to(self.device)
        if not self.regular:
            self.in_msg_buffer = torch.cat([self.in_msg_buffer, self.ps_weight])
        if self.device.type == 'cpu':
            try:
                self.in_msg_buffer = self.in_msg_buffer.pin_memory()
            except Exception as e:
                if self.logger is not None:
                    self.logger.error(e)
                else:
                    raise
        self.placeholder = self.in_msg_buffer.clone()

    @property
    def ps_weight(self) -> torch.Tensor:
        return self._ps_weight

    @ps_weight.setter
    def ps_weight(self, v: torch.Tensor) -> None:
        self._ps_weight.data[0] = v

    @property
    def peers_per_itr(self) -> int:
        return self._graph_manager.peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v: int) -> None:
        self._graph_manager.peers_per_itr = v

    def refresh_peers_(self, rotate: Optional[bool]=None) -> None:
        """Update in- and out-peers"""
        if rotate is None:
            rotate = self._graph_manager.is_dynamic_graph()
        assert not (rotate and (not self._graph_manager.is_dynamic_graph()))
        self.out_edges, self.in_edges = self._graph_manager.get_edges(rotate)

    def refresh_mixing_weights_(self, residual_adjusted: bool=False) -> None:
        """Update mixing-matrix weights"""
        self.mixing_weights = self._mixing_manager.get_mixing_weights(residual_adjusted)

    def mix_out_msg_(self, out_msg: torch.Tensor, ps_weight: torch.Tensor) -> Iterator[torch.Tensor]:
        """Returns a generator mixing messages on the fly"""
        self.refresh_mixing_weights_(residual_adjusted=True)
        self.ps_weight = ps_weight
        if not self.regular:
            out_msg = torch.cat([out_msg, cast(torch.Tensor, self.ps_weight.type(out_msg.dtype))])
        if self._mixing_manager.is_uniform():
            weight = self.mixing_weights['uniform']
            out_msg *= weight.type(out_msg.dtype)
            for _ in self.out_edges:
                yield out_msg
        else:
            for out_edge in self.out_edges:
                weight = self.mixing_weights[out_edge.dest]
                yield out_msg.mul(weight.type(out_msg.dtype))

    def clean_msg_buffers_(self) -> None:
        """Clean outgoing message buffer"""
        while len(self.out_msg_buffer) > 0:
            req, msg = self.out_msg_buffer.pop()
            req.wait()
            msg.set_()

    def parse_in_msg_buffer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse in-msg buffer and return msg and ps-weight separately"""
        msg = self.in_msg_buffer
        if not self.regular:
            return (msg.narrow(0, 0, len(msg) - 1), msg[-1])
        else:
            return (msg, self.ps_weight * self.peers_per_itr_device)

    def mix(self, out_msg: torch.Tensor, ps_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single gossip step"""
        raise NotImplementedError