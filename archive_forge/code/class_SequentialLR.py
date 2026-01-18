import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
class SequentialLR(LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): Does nothing.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose='deprecated'):
        for scheduler_idx in range(len(schedulers)):
            if schedulers[scheduler_idx].optimizer != optimizer:
                raise ValueError(f'Sequential Schedulers expects all schedulers to belong to the same optimizer, but got schedulers at index {scheduler_idx} to be different than the optimizer passed in.')
            if schedulers[scheduler_idx].optimizer != schedulers[0].optimizer:
                raise ValueError(f'Sequential Schedulers expects all schedulers to belong to the same optimizer, but got schedulers at index {0} and {scheduler_idx} to be different.')
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(f'Sequential Schedulers expects number of schedulers provided to be one more than the number of milestone points, but got number of schedulers {len(schedulers)} and the number of milestones to be equal to {len(milestones)}')
        _check_verbose_deprecated_warning(verbose)
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr']
        for scheduler in self._schedulers:
            scheduler.last_epoch -= 1
        self._schedulers[0]._initial_step()
        self._last_lr = schedulers[0].get_last_lr()

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler.step(0)
        else:
            scheduler.step()
        self._last_lr = scheduler.get_last_lr()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)
        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        state_dict['_schedulers'] = _schedulers
        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)