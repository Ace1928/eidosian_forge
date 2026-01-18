import ctypes
import enum
import os
import platform
import sys
import numpy as np
def _get_ops_details(self):
    """Gets op details for every node.

    Returns:
      A list of dictionaries containing arrays with lists of tensor ids for
      tensors involved in the op.
    """
    return [self._get_op_details(idx) for idx in range(self._interpreter.NumNodes())]