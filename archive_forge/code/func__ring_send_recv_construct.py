from typing import cast, Dict, List, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
def _ring_send_recv_construct(in_tensor, d1, d2, left, right, rank, size):
    send_to_right = in_tensor[:, :, :, -d1:].contiguous()
    send_to_left = in_tensor[:, :, :, :d2].contiguous()
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)
    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)
    reqs = dist.batch_isend_irecv([send_op_right, send_op_left, recv_op_left, recv_op_right])
    for req in reqs:
        req.wait()
    if rank == 0:
        in_tensor = torch.cat([in_tensor, recv_from_right], dim=-1)
    elif rank == size - 1:
        in_tensor = torch.cat([recv_from_left, in_tensor], dim=-1)
    else:
        in_tensor = torch.cat([recv_from_left, in_tensor, recv_from_right], dim=-1)
    return in_tensor