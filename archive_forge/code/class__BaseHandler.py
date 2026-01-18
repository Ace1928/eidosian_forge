import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
class _BaseHandler:
    """A base handler for all the passes used here for the SSA algorithm.
    """

    def on_assign(self, states, assign):
        """
        Called when the pass sees an ``ir.Assign``.

        Subclasses should override this for custom behavior

        Parameters
        -----------
        states : dict
        assign : numba.ir.Assign

        Returns
        -------
        stmt : numba.ir.Assign or None
            For rewrite passes, the return value is used as the replacement
            for the given statement.
        """

    def on_other(self, states, stmt):
        """
        Called when the pass sees an ``ir.Stmt`` that's not an assignment.

        Subclasses should override this for custom behavior

        Parameters
        -----------
        states : dict
        assign : numba.ir.Stmt

        Returns
        -------
        stmt : numba.ir.Stmt or None
            For rewrite passes, the return value is used as the replacement
            for the given statement.
        """