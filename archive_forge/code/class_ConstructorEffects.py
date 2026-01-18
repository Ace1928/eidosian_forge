from pythran.analyses.aliases import Aliases
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
class ConstructorEffects(object):

    def __init__(self, node):
        self.func = node
        self.dependencies = lambda ctx: 0
        self.read_effects = [0]