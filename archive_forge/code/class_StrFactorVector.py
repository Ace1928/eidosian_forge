import jinja2  # type: ignore
from rpy2.robjects import (vectors,
from rpy2 import rinterface
from rpy2.robjects.packages import SourceCode
from rpy2.robjects.packages import wherefrom
from IPython import get_ipython  # type: ignore
class StrFactorVector(vectors.FactorVector):

    def __getitem__(self, item):
        integer = super(StrFactorVector, self).__getitem__(item)
        return self.levels[integer - 1]