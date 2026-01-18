from sys import version_info as _swig_python_version_info
import weakref
class TypeRequirementChecker(TypeRegulationsChecker):
    """ Checker for type requirements."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, model):
        _pywrapcp.TypeRequirementChecker_swiginit(self, _pywrapcp.new_TypeRequirementChecker(model))
    __swig_destroy__ = _pywrapcp.delete_TypeRequirementChecker