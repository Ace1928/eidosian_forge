from sys import version_info as _swig_python_version_info
import weakref
class AssignmentElement(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr

    def Activate(self):
        return _pywrapcp.AssignmentElement_Activate(self)

    def Deactivate(self):
        return _pywrapcp.AssignmentElement_Deactivate(self)

    def Activated(self):
        return _pywrapcp.AssignmentElement_Activated(self)
    __swig_destroy__ = _pywrapcp.delete_AssignmentElement