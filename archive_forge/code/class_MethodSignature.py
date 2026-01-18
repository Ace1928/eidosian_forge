import calendar
from typing import Any, Optional, Tuple
class MethodSignature:
    """
    A signature of a callable.
    """

    def __init__(self, *sigList):
        """"""
        self.methodSignature = sigList

    def getArgument(self, name):
        for a in self.methodSignature:
            if a.name == name:
                return a

    def method(self, callable, takesRequest=False):
        return FormMethod(self, callable, takesRequest)