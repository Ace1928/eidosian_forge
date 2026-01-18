from __future__ import print_function, absolute_import
from textwrap import dedent
from shibokensupport.signature import inspect, typing
from shibokensupport.signature.mapping import ellipsis
from shibokensupport.signature.lib.tool import SimpleNamespace
def _attributeerror(self, err_keys):
    err_keys = ', '.join(err_keys)
    allowed_keys = ', '.join(self.allowed_keys.__dict__.keys())
    raise AttributeError(dedent("            Not allowed: '{err_keys}'.\n            The only allowed keywords are '{allowed_keys}'.\n            ".format(**locals())))