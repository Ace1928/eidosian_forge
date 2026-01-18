import torch
from .. import heuristics, jit
from .. import language as tl
from .. import next_power_of_2
class _cross_entropy(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, logits, indices):
        assert indices.dtype == torch.int64, 'Indices are expected to be of type long.'
        device, dtype = (logits.device, logits.dtype)
        n_cols = logits.shape[-1]
        result = torch.empty_like(indices, dtype=dtype, device=device)
        neg_logprobs = torch.empty_like(logits, dtype=dtype, device=device)
        grid = lambda opt: (logits.numel() // n_cols,)
        _forward[grid](logits, neg_logprobs, indices, result, n_cols)
        ctx.save_for_backward(neg_logprobs, indices)
        return result

    @classmethod
    def backward(cls, ctx, dneg_logprobs):
        """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
        neg_logprobs, indices = ctx.saved_tensors
        n_cols = neg_logprobs.shape[-1]
        grid = lambda opt: (neg_logprobs.numel() // n_cols,)
        _backward[grid](neg_logprobs, indices, dneg_logprobs, n_cols)
        return (neg_logprobs, None)