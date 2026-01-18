import calendar
from typing import Any, Optional, Tuple
class FormMethod:
    """A callable object with a signature."""

    def __init__(self, signature, callable, takesRequest=False):
        self.signature = signature
        self.callable = callable
        self.takesRequest = takesRequest

    def getArgs(self):
        return tuple(self.signature.methodSignature)

    def call(self, *args, **kw):
        return self.callable(*args, **kw)