import logging
from math import cos, pi
class MultiFactorScheduler(LRScheduler):
    """Reduce the learning rate by given a list of steps.

    Assume there exists *k* such that::

       step[k] <= num_update and num_update < step[k+1]

    Then calculate the new learning rate by::

       base_lr * pow(factor, k+1)

    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate.
    warmup_steps: int
        number of warmup steps used before this scheduler starts decay
    warmup_begin_lr: float
        if using warmup, the learning rate from which it starts warming up
    warmup_mode: string
        warmup can be done in two modes.
        'linear' mode gradually increases lr with each step in equal increments
        'constant' mode keeps lr at warmup_begin_lr for warmup_steps
    """

    def __init__(self, step, factor=1, base_lr=0.01, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(MultiFactorScheduler, self).__init__(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i - 1]:
                raise ValueError('Schedule step must be an increasing integer list')
            if _step < 1:
                raise ValueError('Schedule step must be greater or equal than 1 round')
        if factor > 1.0:
            raise ValueError('Factor must be no more than 1 to make lr reduce')
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)
        while self.cur_step_ind <= len(self.step) - 1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info('Update[%d]: Change learning rate to %0.5e', num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr