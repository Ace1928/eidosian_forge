import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once
class MemoryEfficientFP16Optimizer(torch.optim.Optimizer):
    """
    Wrap an optimizer to perform memory-efficient mixed precision training.

    This class wraps an optimizer to perform FP16 training.
    This implementation is heavily based on the Fairseq implementation
    of `MemoryEfficientFP16Optimizer`, which can be found here:
    <https://github.com/pytorch/fairseq/blob/master/fairseq/optim/fp16_optimizer.py#L382>

    This allows you to train bigger models on a single GPU, but can be unstable.
    Opt for the APEX implementation if you do not have concerns about memory.

    :param params:
        Model parameters
    :param optimizer:
        Any torch optimizer
    :param float loss_initial_scale:
        Initial loss scaling. Default chosen empirically, but models with very low
        or high loss values may need this adjusted. Stick with powers of 2
    :param float min_loss_scale:
        Throws an error if your loss scale goes below this threshold
    """

    def __init__(self, init_optimizer: torch.optim.Optimizer, loss_initial_scale: float=2.0 ** 17, min_loss_scale: float=0.0001):
        self.optimizer = init_optimizer
        self.min_loss_scale = min_loss_scale
        self.scaler = DynamicLossScaler(init_scale=loss_initial_scale)

    @staticmethod
    def compatible_optimizers():
        """
        List of compatible optimizers.
        """
        return ['adam', 'mem_eff_adam', 'adafactor']

    @property
    def params(self):
        """
        Return an iterable of the parameters held by the optimizer.
        """
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)

    def __repr__(self):
        self.optimizer.__repr__()

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def _unscale_grads(self, multiply_grads=1.0):
        if self._grads_are_scaled:
            self._grads_are_scaled = False
            self.multiply_grads(multiply_grads / self.scaler.loss_scale)
        else:
            assert multiply_grads == 1.0

    def clip_master_grads(self, gradient_clip):
        """
        Clips gradient norm and updates dynamic loss scaler.

        Returns -1 if the most recently computed gradients overflowed.
        """
        self._unscale_grads()
        grad_norm = clip_grad_norm(self.params, gradient_clip)
        overflow = has_overflow(grad_norm)
        self.scaler.update_scale(overflow)
        if overflow:
            if self.scaler.loss_scale <= self.min_loss_scale:
                raise FloatingPointError('Minimum loss scale reached ({}). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.'.format(self.min_loss_scale))
            logging.info(f'Overflow: setting loss scale to {self.scaler.loss_scale}')
            self.zero_grad()
            return -1
        return grad_norm

    def update_master_grads(self):
        pass

    def multiply_grads(self, c):
        """
        Multiplies grads by a constant `c`.
        """
        if self._grads_are_scaled:
            self._unscale_grads(c)
        else:
            for p in self.params:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def backward(self, loss, update_master_grads=False):
        """
        Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to a regular backwards call , this function dynamically scales the loss
        to avoid gradient underflow.
        """
        loss = loss * self.scaler.loss_scale
        loss.backward()
        self._grads_are_scaled = True

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        self._unscale_grads()
        self.optimizer.step(closure)

    def state_dict(self):
        """
        Return the optimizer's state dict.
        """
        state_dict = self.optimizer.state_dict()
        state_dict['loss_scaler'] = self.scaler
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Load an optimizer state dict.

        Override from PyTorch implementation to avoid casting to FP32.
        """
        if 'loss_scaler' in state_dict:
            self.scaler = state_dict['loss_scaler']
        self.optimizer.load_state_dict(state_dict)
        groups = self.optimizer.param_groups
        saved_groups = state_dict['param_groups']
        id_map = {old_id: p for old_id, p in zip(chain(*(g['params'] for g in saved_groups)), chain(*(g['params'] for g in groups)))}
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                like_device_v = {j: w.to(param.device) if torch.is_tensor(w) else w for j, w in v.items()}
                self.optimizer.state[param] = like_device_v

    @property
    def loss_scale(self):
        """
        Convenience function which TorchAgent calls to get current scale value.
        """
        return self.scaler.loss_scale

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()
        self._grads_are_scaled = False