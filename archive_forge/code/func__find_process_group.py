import logging
import warnings
from collections import OrderedDict
from typing import Union, Iterable, Dict
import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.utils as utils
def _find_process_group(self):
    """
        Returns a process group as the value of an ``period_process_group_dict`` entry,
        if ``step`` can be divided by a period in the keys of ``period_process_group_dict``.
        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        then the returned process group is the one corresponding to the largest period,
        since this process group will be used for averaging parameters at this ``step``.
        Returns ``None`` if not found.
        """
    for period in reversed(self._periods):
        if self.step % period == 0:
            return self.period_process_group_dict[period]
    return None