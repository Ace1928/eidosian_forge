from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
class Optimizer2State(Optimizer8bit):

    def __init__(self, optimizer_name, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, optim_bits=32, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, max_unorm=0.0, skip_zeros=False, is_paged=False):
        """
        Base 2-state update optimizer class.

        Arguments:
            optimizer_name (`str`):
                The name of the optimizer.
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple`, defaults to (0.9, 0.999)):
                The beta values for the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value for the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`dict`, defaults to `None`):
                A dictionary with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            max_unorm (`float`, defaults to 0.0):
                The maximum value to normalize each block with.
            skip_zeros (`bool`, defaults to `False`):
                Whether to skip zero values for sparse gradients and models to ensure correct updates.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if isinstance(betas, str):
            betas = betas.replace('(', '').replace(')', '').strip().split(',')
            betas = [float(b) for b in betas]
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(f'Invalid beta parameter at index {i}: {betas[i]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults, optim_bits, is_paged)
        if args is None:
            args = {}
            args['optim_bits'] = optim_bits
            args['percentile_clipping'] = 100
            args['min_8bit_size'] = min_8bit_size
            args['percentile_clipping'] = percentile_clipping
            args['block_wise'] = block_wise
            args['max_unorm'] = max_unorm
            args['skip_zeros'] = skip_zeros
            self.args = MockArgs(args)
        else:
            self.args = args
        self.optimizer_name = optimizer_name

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)
        if config['optim_bits'] == 32:
            dtype = torch.float32
        elif config['optim_bits'] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(f'Amount of optimizer bits not supported: {config['optim_bits']}')
        if p.numel() < config['min_8bit_size']:
            dtype = torch.float32
        state = self.state[p]
        state['step'] = 0
        if dtype == torch.float32 or (dtype == torch.uint8 and p.numel() < 4096):
            state['state1'] = self.get_state_buffer(p, dtype=torch.float32)
            state['state2'] = self.get_state_buffer(p, dtype=torch.float32)
        elif dtype == torch.uint8:
            if state['step'] == 0:
                if 'dynamic' not in self.name2qmap:
                    self.fill_qmap()
                self.name2qmap['dynamic'] = self.name2qmap['dynamic'].to(p.device)
                self.name2qmap['udynamic'] = self.name2qmap['udynamic'].to(p.device)
            state['state1'] = self.get_state_buffer(p, dtype=torch.uint8)
            state['qmap1'] = self.name2qmap['dynamic']
            state['state2'] = self.get_state_buffer(p, dtype=torch.uint8)
            state['qmap2'] = self.name2qmap['udynamic']
            if config['block_wise']:
                n = p.numel()
                blocks = n // 2048
                blocks += 1 if n % 2048 > 0 else 0
                state['absmax1'] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
                state['absmax2'] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
            else:
                state['max1'] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state['new_max1'] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state['max2'] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state['new_max2'] = torch.zeros((1,), dtype=torch.float32, device=p.device)
        if config['percentile_clipping'] < 100:
            state['gnorm_vec'] = torch.zeros((100,), device=p.device)
        if config['max_unorm'] > 0.0:
            state['unorm_vec'] = torch.zeros((1,), device=p.device)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        state = self.state[p]
        grad = p.grad
        config = self.get_config(gindex, pindex, group)
        state['step'] += 1
        step = state['step']
        if config['percentile_clipping'] < 100:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(grad, state['gnorm_vec'], step, config['percentile_clipping'])
        else:
            gnorm_scale = 1.0
        if state['state1'].dtype == torch.float:
            F.optimizer_update_32bit(self.optimizer_name, grad, p, state['state1'], config['betas'][0], config['eps'], step, config['lr'], state['state2'], config['betas'][1], config['weight_decay'], gnorm_scale, state['unorm_vec'] if config['max_unorm'] > 0.0 else None, max_unorm=config['max_unorm'], skip_zeros=config['skip_zeros'])
        elif state['state1'].dtype == torch.uint8 and (not config['block_wise']):
            F.optimizer_update_8bit(self.optimizer_name, grad, p, state['state1'], state['state2'], config['betas'][0], config['betas'][1], config['eps'], step, config['lr'], state['qmap1'], state['qmap2'], state['max1'], state['max2'], state['new_max1'], state['new_max2'], config['weight_decay'], gnorm_scale=gnorm_scale, unorm_vec=state['unorm_vec'] if config['max_unorm'] > 0.0 else None, max_unorm=config['max_unorm'])
            state['max1'], state['new_max1'] = (state['new_max1'], state['max1'])
            state['max2'], state['new_max2'] = (state['new_max2'], state['max2'])
        elif state['state1'].dtype == torch.uint8 and config['block_wise']:
            F.optimizer_update_8bit_blockwise(self.optimizer_name, grad, p, state['state1'], state['state2'], config['betas'][0], config['betas'][1], config['eps'], step, config['lr'], state['qmap1'], state['qmap2'], state['absmax1'], state['absmax2'], config['weight_decay'], gnorm_scale=gnorm_scale, skip_zeros=config['skip_zeros'])