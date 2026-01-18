import logging
from math import cos, pi
class PolyScheduler(LRScheduler):
    """ Reduce the learning rate according to a polynomial of given power.

    Calculate the new learning rate, after warmup if any, by::

       final_lr + (start_lr - final_lr) * (1-nup/max_nup)^pwr
       if nup < max_nup, 0 otherwise.

    Parameters
    ----------
        max_update: int
            maximum number of updates before the decay reaches final learning rate.
        base_lr: float
            base learning rate to start from
        pwr:   int
            power of the decay term as a function of the current number of updates.
        final_lr:   float
            final learning rate after all steps
        warmup_steps: int
            number of warmup steps used before this scheduler starts decay
        warmup_begin_lr: float
            if using warmup, the learning rate from which it starts warming up
        warmup_mode: string
            warmup can be done in two modes.
            'linear' mode gradually increases lr with each step in equal increments
            'constant' mode keeps lr at warmup_begin_lr for warmup_steps
    """

    def __init__(self, max_update, base_lr=0.01, pwr=2, final_lr=0, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(PolyScheduler, self).__init__(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        assert isinstance(max_update, int)
        if max_update < 1:
            raise ValueError('maximum number of updates must be strictly positive')
        self.power = pwr
        self.base_lr_orig = self.base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.max_steps = self.max_update - self.warmup_steps

    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)
        if num_update <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * pow(1 - float(num_update - self.warmup_steps) / float(self.max_steps), self.power)
        return self.base_lr