from typing import Tuple, Optional, Union
import torch
from einops import rearrange
import triton
import triton.language as tl
class CrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignored_index=-100, inplace_backward=False, process_group=None):
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        total_classes = world_size * n_cols
        rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
        class_start_idx = rank * n_cols
        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        MAX_BLOCK_SIZE = 64 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
        num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 128 * 1024 else 32
        split = world_size > 1 or n_cols > MAX_BLOCK_SIZE
        n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
        losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        with torch.cuda.device(logits.device.index):
            cross_entropy_fwd_kernel[n_rows, n_splits](losses, lse, z_losses, logits, labels, smoothing, logit_scale, lse_square_scale, ignored_index, total_classes, class_start_idx, n_cols, n_rows, logits.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, SPLIT=split)
        if split:
            if n_splits > 1:
                lse = torch.logsumexp(lse, dim=0)
                losses = losses.sum(dim=0)
            if world_size > 1:
                lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
                torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
                handle_losses = torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True)
                lse = torch.logsumexp(lse_allgather, dim=0)
                handle_losses.wait()
            losses += lse
            if lse_square_scale != 0.0:
                z_losses = lse_square_scale * lse.square()
                z_losses.masked_fill_(labels == ignored_index, 0.0)
                losses += z_losses
            else:
                z_losses = torch.zeros_like(losses)
            losses.masked_fill_(labels == ignored_index, 0.0)
        ctx.save_for_backward(logits, lse, labels)
        ctx.mark_non_differentiable(z_losses)
        ctx.smoothing = smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward
        return (losses, z_losses)

    @staticmethod
    def backward(ctx, grad_losses, grad_z_losses):
        del grad_z_losses
        logits, lse, labels = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16
        grid = lambda META: (n_rows, triton.cdiv(n_cols, META['BLOCK_SIZE']))
        with torch.cuda.device(logits.device.index):
            cross_entropy_bwd_kernel[grid](dlogits, grad_losses, logits, lse, labels, ctx.smoothing, ctx.logit_scale, ctx.lse_square_scale, ctx.ignored_index, ctx.total_classes, ctx.class_start_idx, n_cols, logits.stride(0), dlogits.stride(0), grad_losses.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        return (dlogits, None, None, None, None, None, None, None, None)