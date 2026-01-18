from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.fp16 import fp16_optimizer_wrapper
from parlai.utils.torch import neginf
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
class AdamWithDecay(Optimizer):
    """
    Adam with decay; mirror's HF's implementation.

    :param lr:
        learning rate
    :param b1:
        Adams b1. Default: 0.9
    :param b2:
        Adams b2. Default: 0.999
    :param e:
        Adams epsilon. Default: 1e-6
    :param weight_decay:
        Weight decay. Default: 0.01
    :param max_grad_norm:
        Maximum norm for the gradients (-1 means no clipping).  Default: 1.0
    """

    def __init__(self, params, lr, b1=0.9, b2=0.999, e=1e-06, weight_decay=0.01, max_grad_norm=1.0):
        if lr < 0.0:
            raise ValueError('Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if not 0.0 <= b1 < 1.0:
            raise ValueError('Invalid b1 parameter: {} - should be in [0.0, 1.0['.format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError('Invalid b2 parameter: {} - should be in [0.0, 1.0['.format(b2))
        if not e >= 0.0:
            raise ValueError('Invalid epsilon value: {} - should be >= 0.0'.format(e))
        defaults = dict(lr=lr, b1=b1, b2=b2, e=e, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(AdamWithDecay, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        :param closure:
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['next_m'] = torch.zeros_like(p.data)
                    state['next_v'] = torch.zeros_like(p.data)
                next_m, next_v = (state['next_m'], state['next_v'])
                beta1, beta2 = (group['b1'], group['b2'])
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data
                lr = group['lr']
                update_with_lr = lr * update
                p.data.add_(-update_with_lr)
        return loss