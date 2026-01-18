from pythran.passmanager import ModuleAnalysis
import pythran.metadata as md
import beniget
class DefUseChains(ModuleAnalysis):
    """
    Build define-use-define chains analysis for each variable.
    """

    def __init__(self):
        self.result = None
        super(DefUseChains, self).__init__()

    def visit_Module(self, node):
        duc = ExtendedDefUseChains()
        duc.visit(node)
        self.result = duc