from bitsandbytes.optim.optimizer import Optimizer2State
class AdamW32bit(Optimizer2State):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, optim_bits=32, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        """
        32-bit AdamW optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
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
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__('adam', params, lr, betas, eps, weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)