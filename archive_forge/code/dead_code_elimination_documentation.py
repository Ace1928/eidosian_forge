from pythran.analyses import PureExpressions, DefUseChains, Ancestors
from pythran.openmp import OMPDirective
from pythran.passmanager import Transformation
import pythran.metadata as metadata
import gast as ast
 Add OMPDirective from the old node to the new one. 