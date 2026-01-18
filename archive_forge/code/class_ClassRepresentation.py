import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
class ClassRepresentation(RS4):
    """ Definition of an R S4 class """
    slots = property(lambda x: [y[0] for y in x.do_slot('slots')], None, None, 'Slots (attributes) for the class')
    basenames = property(lambda x: [y[0] for y in x.do_slot('contains')], None, None, 'Parent classes')
    contains = basenames
    isabstract = property(lambda x: x.do_slot('virtual')[0], None, None, 'Is the class an abstract class ?')
    virtual = isabstract
    packagename = property(lambda x: x.do_slot('package')[0], None, None, 'R package in which the class is defined')
    package = packagename
    classname = property(lambda x: x.do_slot('className')[0], None, None, 'Name of the R class')