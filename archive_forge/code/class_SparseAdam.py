import torch
from . import _functional as F
from .optimizer import Optimizer, _maximize_doc
class SparseAdam(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, maximize: bool=False):
        if not 0.0 < lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 < eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        defaults = dict(lr=lr, betas=betas, eps=eps, maximize=maximize)
        super().__init__(params, defaults)
        sparse_params = []
        for index, param_group in enumerate(self.param_groups):
            assert isinstance(param_group, dict), f'param_groups must be a list of dicts, but got {type(param_group)}'
            for d_index, d_param in enumerate(param_group['params']):
                if d_param.is_sparse:
                    sparse_params.append([index, d_index])
        if sparse_params:
            raise ValueError(f'Sparse params at indices {sparse_params}: SparseAdam requires dense parameter tensors')

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            eps = group['eps']
            lr = group['lr']
            beta1, beta2 = group['betas']
            maximize = group.get('maximize', False)
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if not p.grad.is_sparse:
                        raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state['step'] += 1
                    state_steps.append(state['step'])
            F.sparse_adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps, beta1=beta1, beta2=beta2, lr=group['lr'], eps=group['eps'], maximize=maximize)
        return loss