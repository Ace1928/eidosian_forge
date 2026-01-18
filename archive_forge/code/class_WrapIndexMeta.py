import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
class WrapIndexMeta(object):
    """
      Array analysis should be able to analyze all the function
      calls that it adds to the IR.  That way, array analysis can
      be run as often as needed and you should get the same
      equivalencies.  One modification to the IR that array analysis
      makes is the insertion of wrap_index calls.  Thus, repeated
      array analysis passes should be able to analyze these wrap_index
      calls.  The difficulty of these calls is that the equivalence
      class of the left-hand side of the assignment is not present in
      the arguments to wrap_index in the right-hand side.  Instead,
      the equivalence class of the wrap_index output is a combination
      of the wrap_index args.  The important thing to
      note is that if the equivalence classes of the slice size
      and the dimension's size are the same for two wrap index
      calls then we can be assured of the answer being the same.
      So, we maintain the wrap_map dict that maps from a tuple
      of equivalence class ids for the slice and dimension size
      to some new equivalence class id for the output size.
      However, when we are analyzing the first such wrap_index
      call we don't have a variable there to associate to the
      size since we're in the process of analyzing the instruction
      that creates that mapping.  So, instead we return an object
      of this special class and analyze_inst will establish the
      connection between a tuple of the parts of this object
      below and the left-hand side variable.
    """

    def __init__(self, slice_size, dim_size):
        self.slice_size = slice_size
        self.dim_size = dim_size