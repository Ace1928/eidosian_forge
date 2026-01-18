from pythran.analyses import LazynessAnalysis, UseDefChains, DefUseChains
from pythran.analyses import Literals, Ancestors, Identifiers, CFG, IsAssigned
from pythran.passmanager import Transformation
import pythran.graph as graph
from collections import defaultdict
import gast as ast
 Satisfy dependencies on others analyses. 