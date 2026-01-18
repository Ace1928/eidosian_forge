from __future__ import print_function, absolute_import
from textwrap import dedent
from shibokensupport.signature import inspect, typing
from shibokensupport.signature.mapping import ellipsis
from shibokensupport.signature.lib.tool import SimpleNamespace
class SignatureLayout(SimpleNamespace):
    """
    Configure a signature.

    The layout of signatures can have different layouts which are
    controlled by keyword arguments:

    definition=True         Determines if self will generated.
    defaults=True
    ellipsis=False          Replaces defaults by "...".
    return_annotation=True
    parameter_names=True    False removes names before ":".
    """
    allowed_keys = SimpleNamespace(definition=True, defaults=True, ellipsis=False, return_annotation=True, parameter_names=True)
    allowed_values = (True, False)

    def __init__(self, **kwds):
        args = SimpleNamespace(**self.allowed_keys.__dict__)
        args.__dict__.update(kwds)
        self.__dict__.update(args.__dict__)
        err_keys = list(set(self.__dict__) - set(self.allowed_keys.__dict__))
        if err_keys:
            self._attributeerror(err_keys)
        err_values = list(set(self.__dict__.values()) - set(self.allowed_values))
        if err_values:
            self._valueerror(err_values)

    def __setattr__(self, key, value):
        if key not in self.allowed_keys.__dict__:
            self._attributeerror([key])
        if value not in self.allowed_values:
            self._valueerror([value])
        self.__dict__[key] = value

    def _attributeerror(self, err_keys):
        err_keys = ', '.join(err_keys)
        allowed_keys = ', '.join(self.allowed_keys.__dict__.keys())
        raise AttributeError(dedent("            Not allowed: '{err_keys}'.\n            The only allowed keywords are '{allowed_keys}'.\n            ".format(**locals())))

    def _valueerror(self, err_values):
        err_values = ', '.join(map(str, err_values))
        allowed_values = ', '.join(map(str, self.allowed_values))
        raise ValueError(dedent("            Not allowed: '{err_values}'.\n            The only allowed values are '{allowed_values}'.\n            ".format(**locals())))