from collections import OrderedDict
from rpy2.robjects.packages import _loaded_namespaces
from rpy2.robjects.vectors import IntVector, FloatVector, ComplexVector
from rpy2.robjects.vectors import Array, Matrix
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.vectors import ListVector, DataFrame
from rpy2.robjects.environments import Environment
from rpy2.rinterface import NULL
from rpy2.robjects import Formula, RS4
from rpy2.robjects import methods
from rpy2.robjects import conversion
from rpy2.robjects import help as rhelp
from rpy2.robjects.language import eval
from . import process_revents as revents
from os import linesep
import re
def __setattr__(self, name, value):
    raise AttributeError("Attributes cannot be set. Use 'importr'")