from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
class PushPull(Gossiper):
    """Doubly-stochastic consensus averaging module"""

    def mix(self, out_msg: torch.Tensor, ps_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'.format(self.in_edges, self.out_edges))
        mixed_out_msgs = self.mix_out_msg_(out_msg, ps_weight)
        if len(self.in_edges) == 1 and len(self.out_edges) == 1:
            out_edge, in_edge = (self.out_edges[0], self.in_edges[0])
            msg = next(mixed_out_msgs)
            if not self.passive:
                dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group)
                dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src, group=in_edge.process_group)
            else:
                dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src, group=in_edge.process_group)
                dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group)
        else:
            self.in_msg_buffer.zero_()
            for out_edge, in_edge in zip(self.out_edges, self.in_edges):
                msg = next(mixed_out_msgs)
                if not self.passive:
                    dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group)
                    dist.broadcast(tensor=self.placeholder, src=in_edge.src, group=in_edge.process_group)
                else:
                    dist.broadcast(tensor=self.placeholder, src=in_edge.src, group=in_edge.process_group)
                    dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group)
                self.in_msg_buffer.add_(self.placeholder)
        self.refresh_peers_()
        self.clean_msg_buffers_()
        return self.parse_in_msg_buffer()