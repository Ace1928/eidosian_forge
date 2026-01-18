import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
class TorchHistory:
    """History methods specific to PyTorch"""

    def __init__(self):
        global torch
        torch = wandb.util.get_module('torch', 'Could not import torch')
        self._hook_handles = {}
        self._num_bins = 64
        self._is_cuda_histc_supported = None
        self.hook_torch = TorchGraph.hook_torch

    def add_log_parameters_hook(self, module: 'torch.nn.Module', name: str='', prefix: str='', log_freq: int=0) -> None:
        """This instruments hooks into the pytorch module
        log parameters after a forward pass
        log_freq - log gradients/parameters every N batches
        """
        prefix = prefix + name
        if not hasattr(module, '_wandb_hook_names'):
            module._wandb_hook_names = []

        def parameter_log_hook(module, input_, output, log_track):
            if not log_track_update(log_track):
                return
            for name, parameter in module.named_parameters():
                if isinstance(parameter, torch.autograd.Variable):
                    data = parameter.data
                else:
                    data = parameter
                self.log_tensor_stats(data.cpu(), 'parameters/' + prefix + name)
        log_track_params = log_track_init(log_freq)
        try:
            hook = module.register_forward_hook(lambda mod, inp, outp: parameter_log_hook(mod, inp, outp, log_track_params))
            self._hook_handles['parameters/' + prefix] = hook
            module._wandb_hook_names.append('parameters/' + prefix)
        except RuntimeError as e:
            wandb.termwarn(f'Trying to register forward_hook failed ({e}) - skipping parameter tracking.')

    def add_log_gradients_hook(self, module: 'torch.nn.Module', name: str='', prefix: str='', log_freq: int=0) -> None:
        """This instruments hooks into the pytorch module
        log gradients after a backward pass
        log_freq - log gradients/parameters every N batches
        """
        prefix = prefix + name
        if not hasattr(module, '_wandb_hook_names'):
            module._wandb_hook_names = []
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                log_track_grad = log_track_init(log_freq)
                module._wandb_hook_names.append('gradients/' + prefix + name)
                self._hook_variable_gradient_stats(parameter, 'gradients/' + prefix + name, log_track_grad)

    def log_tensor_stats(self, tensor, name):
        """Add distribution statistics on a tensor's elements to the current History entry"""
        if isinstance(tensor, (tuple, list)):
            while isinstance(tensor, (tuple, list)) and isinstance(tensor[0], (tuple, list)):
                tensor = [item for sublist in tensor for item in sublist]
            tensor = torch.cat([t.detach().clone().reshape(-1) for t in tensor])
        tensor = tensor.detach().clone()
        if not hasattr(tensor, 'shape'):
            cls = type(tensor)
            raise TypeError(f'Expected Tensor, not {cls.__module__}.{cls.__name__}')
        sparse_zeros = None
        if tensor.is_sparse:
            tensor = tensor.cpu().coalesce()
            backing_values = tensor._values()
            sparse_zeros = tensor.numel() - backing_values.numel()
            tensor = backing_values
        flat = tensor.reshape(-1)
        if flat.is_cuda:
            if self._is_cuda_histc_supported is None:
                try:
                    flat.histc(bins=self._num_bins)
                except RuntimeError:
                    self._is_cuda_histc_supported = False
                else:
                    self._is_cuda_histc_supported = True
            if not self._is_cuda_histc_supported:
                flat = flat.cpu()
            elif not isinstance(flat, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
                flat = flat.type(torch.cuda.FloatTensor)
        if not flat.is_cuda and (not isinstance(flat, (torch.FloatTensor, torch.DoubleTensor))):
            flat = flat.type(torch.FloatTensor)
        if self._no_finite_values(flat):
            return
        flat = self._remove_infs_nans(flat)
        tmin = flat.min().item()
        tmax = flat.max().item()
        if sparse_zeros:
            tmin = 0 if tmin > 0 else tmin
            tmax = 0 if tmax < 0 else tmax
        if tmin > tmax:
            tmin, tmax = (tmax, tmin)
        if tmin == tmax:
            tensor = torch.Tensor([flat.numel()])
            tensor = tensor.cpu().clone().detach()
            bins = torch.Tensor([tmin, tmax])
        else:
            tensor = flat.histc(bins=self._num_bins, min=tmin, max=tmax)
            tensor = tensor.cpu().detach().clone()
            bins = torch.linspace(tmin, tmax, steps=self._num_bins + 1)
        if sparse_zeros:
            bins_np = bins.numpy()
            tensor_np = tensor.numpy()
            bin_idx = 0
            num_buckets = len(bins_np) - 1
            for i in range(num_buckets):
                start = bins_np[i]
                end = bins_np[i + 1]
                if start <= 0 and end > 0 or (i == num_buckets - 1 and end == 0):
                    bin_idx = i
                    break
            tensor_np[bin_idx] += sparse_zeros
            tensor = torch.Tensor(tensor_np)
            bins = torch.Tensor(bins_np)
        wandb.run._log({name: wandb.Histogram(np_histogram=(tensor.tolist(), bins.tolist()))}, commit=False)

    def _hook_variable_gradient_stats(self, var, name, log_track):
        """Logs a Variable's gradient's distribution statistics next time backward()
        is called on it.
        """
        if not isinstance(var, torch.autograd.Variable):
            cls = type(var)
            raise TypeError(f'Expected torch.Variable, not {cls.__module__}.{cls.__name__}')
        handle = self._hook_handles.get(name)
        if handle is not None and self._torch_hook_handle_is_valid(handle):
            raise ValueError(f'A hook has already been set under name "{name}"')

        def _callback(grad, log_track):
            if not log_track_update(log_track):
                return
            self.log_tensor_stats(grad.data, name)
        handle = var.register_hook(lambda grad: _callback(grad, log_track))
        self._hook_handles[name] = handle
        return handle

    def unhook_all(self):
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles = {}

    def unhook(self, name):
        handle = self._hook_handles.pop(name)
        handle.remove()

    def _torch_hook_handle_is_valid(self, handle):
        d = handle.hooks_dict_ref()
        if d is None:
            return False
        else:
            return handle.id in d

    def _no_finite_values(self, tensor: 'torch.Tensor') -> bool:
        return tensor.shape == torch.Size([0]) or (~torch.isfinite(tensor)).all().item()

    def _remove_infs_nans(self, tensor: 'torch.Tensor') -> 'torch.Tensor':
        if not torch.isfinite(tensor).all():
            tensor = tensor[torch.isfinite(tensor)]
        return tensor