import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def _normalize_asm(self, asm):
    asm = textwrap.dedent(asm)
    asm = asm.replace('\n    ', '\n  ')
    return asm