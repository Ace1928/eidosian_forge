import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
@pytest.fixture(scope='module')
def _set_class_AB():
    robjects.r('A <- methods::setClass("A", representation(x="integer"))')
    robjects.r('B <- methods::setClass("B", contains="A")')
    yield
    robjects.r('methods::removeClass("B")')
    robjects.r('methods::removeClass("A")')