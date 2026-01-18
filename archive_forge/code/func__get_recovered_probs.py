from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
def _get_recovered_probs(self, target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    """Create a probability distribution for each proposed token which can
        be sampled if the proposed token is rejected.

        When this routine is applied sequentially, the true distribution of the
        target model is recovered (within hardware numerics).

        The probability distribution used in this rejection case is constructed
        as follows. Given :math:`q(x|x_1, \\dots, x_n)`, the probability of
        :math:`x` given context :math:`x_1, \\dots, x_n` according to the target
        model and :math:`p(x|x_1, \\dots, x_n)`, the same conditional probability
        according to the draft model:

        .. math::
            x_{n+1} \\sim (q(x|x_1, \\dots, x_n) - p(x|x_1, \\dots, x_n))_+

        where :math:`(f(x))_+` is defined as:

        .. math::
            (f(x))_+ = \\frac{\\max(0, f(x))}{\\sum_x \\max(0, f(x))}

        See https://github.com/vllm-project/vllm/pull/2336 for a visualization
        of the draft, target, and recovered probability distributions.

        Returns a tensor of shape [batch_size, k, vocab_size].

        Note: This batches operations on GPU and thus constructs the recovered
        distribution for all tokens, even if they are accepted. This causes
        division-by-zero errors, so we use self._smallest_positive_value to
        avoid that. This introduces some drift to the distribution.
        """
    _, k, _ = draft_probs.shape
    difference = target_probs - draft_probs
    f = torch.clamp(difference, min=self._smallest_positive_value)
    recovered_probs = f / torch.sum(f, dim=-1).reshape(-1, k, 1)
    return recovered_probs