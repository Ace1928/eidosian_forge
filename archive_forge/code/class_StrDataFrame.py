import jinja2  # type: ignore
from rpy2.robjects import (vectors,
from rpy2 import rinterface
from rpy2.robjects.packages import SourceCode
from rpy2.robjects.packages import wherefrom
from IPython import get_ipython  # type: ignore
class StrDataFrame(vectors.DataFrame):

    def __getitem__(self, item):
        obj = super(StrDataFrame, self).__getitem__(item)
        if isinstance(obj, vectors.FactorVector):
            obj = StrFactorVector(obj)
        return obj