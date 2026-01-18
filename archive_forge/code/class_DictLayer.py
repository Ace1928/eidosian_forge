from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.intrinsic import Intrinsic, Class, UnboundValue
from pythran.passmanager import ModuleAnalysis
from pythran.tables import functions, methods, MODULES
from pythran.unparse import Unparser
from pythran.conversion import demangle
import pythran.metadata as md
from pythran.utils import isnum
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
from itertools import product
import io
class DictLayer(object):

    def __init__(self, base, layer=None):
        self.base = base
        self.layer = dict() if layer is None else layer

    def __getitem__(self, key):
        return (self.layer if key in self.layer else self.base)[key]

    def __contains__(self, key):
        return key in self.layer or key in self.base

    def __setitem__(self, key, value):
        self.layer[key] = value

    def get(self, key, default=None):
        return self.layer.get(key, self.base.get(key, default))