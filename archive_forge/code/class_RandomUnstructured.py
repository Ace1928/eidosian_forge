import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
class RandomUnstructured(BasePruningMethod):
    """Prune (currently unpruned) units in a tensor at random.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        _validate_pruning_amount(nparams_toprune, tensor_size)
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if nparams_toprune != 0:
            prob = torch.rand_like(t)
            topk = torch.topk(prob.view(-1), k=nparams_toprune)
            mask.view(-1)[topk.indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super().apply(module, name, amount=amount)