import functools
import torch
import torch.cuda
from torch.testing._internal.common_utils import LazyVal, TEST_NUMBA, TEST_WITH_ROCM, TEST_CUDA, IS_WINDOWS
import inspect
import contextlib
def _create_scaling_models_optimizers(device='cuda', optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    with torch.no_grad():
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.copy_(c)
    kwargs = {'lr': 1.0}
    if optimizer_kwargs is not None:
        kwargs.update(optimizer_kwargs)
    opt_control = optimizer_ctor(mod_control.parameters(), **kwargs)
    opt_scaling = optimizer_ctor(mod_scaling.parameters(), **kwargs)
    return (mod_control, mod_scaling, opt_control, opt_scaling)