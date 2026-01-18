from typing import Any, Dict, Iterable, List, no_type_check, Type
import torch

    Return a list of in-backward optimizers applied to ``module``'s parameters. Note that these
    optimizers are not intended to directly have their ``step`` or ``zero_grad`` methods called
    by the user and are intended to be used for things like checkpointing.

    Args:
        module: (torch.nn.Module): model to retrieve in-backward optimizers for

    Returns:
        List[torch.optim.Optimizer]: the in-backward optimizers.

    Example::
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {'lr': 0.01})
        optims = _get_optimizers_in_backward(model)
    