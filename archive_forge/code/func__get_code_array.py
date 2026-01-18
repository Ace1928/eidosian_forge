import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _get_code_array(co, adaptive):
    return co._co_code_adaptive if adaptive else co.co_code