from pythran.passmanager import Transformation
from pythran.utils import path_to_attr, path_to_node
from pythran.conversion import mangle
from pythran.syntax import PythranSyntaxError
from pythran.analyses import Ancestors
import gast as ast

        Replace name with full expanded name.

        Examples
        --------
        >> from numpy.linalg import det

        >> det(a)

        Becomes

        >> numpy.linalg.det(a)
        