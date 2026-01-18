import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
def _get_timer_from_stack(self, stack):
    """
        This method gets the timer associated with stack.

        Parameters
        ----------
        stack: list of str
            A list of identifiers.

        Returns
        -------
        timer: _HierarchicalHelper
        """
    tmp = self
    for i in stack:
        tmp = tmp.timers[i]
    return tmp