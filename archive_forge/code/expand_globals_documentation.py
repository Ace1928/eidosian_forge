from pythran.analyses import LocalNameDeclarations
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.utils import path_to_attr
from pythran import metadata
import gast as ast

        Turn global variable used not shadows to function call.

        We check it is a name from an assignment as import or functions use
        should not be turn into call.
        