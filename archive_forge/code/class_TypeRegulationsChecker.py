from sys import version_info as _swig_python_version_info
import weakref
class TypeRegulationsChecker(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined - class is abstract')
    __repr__ = _swig_repr
    __swig_destroy__ = _pywrapcp.delete_TypeRegulationsChecker

    def CheckVehicle(self, vehicle, next_accessor):
        return _pywrapcp.TypeRegulationsChecker_CheckVehicle(self, vehicle, next_accessor)