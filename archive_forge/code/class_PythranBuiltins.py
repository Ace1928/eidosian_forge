from pythran.analyses import ConstantExpressions, ASTMatcher
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import to_ast, ConversionError, ToNotEval, mangle
from pythran.analyses.ast_matcher import DamnTooLongPattern
from pythran.syntax import PythranSyntaxError
from pythran.utils import isintegral, isnum
from pythran.config import cfg
import builtins
import gast as ast
from copy import deepcopy
import logging
import sys
class PythranBuiltins(object):

    @staticmethod
    def static_list(*args):
        return list(*args)

    @staticmethod
    def static_if(cond, true_br, false_br):
        return true_br if cond else false_br

    @staticmethod
    def is_none(val):
        return val is None

    @staticmethod
    def make_shape(*args):
        return args