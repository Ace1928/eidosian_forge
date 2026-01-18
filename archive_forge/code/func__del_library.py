from ._ops import OpOverload
from typing import Any, Optional, Set, List
import traceback
import torch
import weakref
import functools
import inspect
import re
import sys
def _del_library(captured_impls, op_impls, captured_defs, op_defs, registration_handles):
    captured_impls -= op_impls
    captured_defs -= op_defs
    for handle in registration_handles:
        handle.destroy()