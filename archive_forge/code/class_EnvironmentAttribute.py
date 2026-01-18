import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class EnvironmentAttribute(Expr):
    """Loads an attribute from the environment object.  This is useful for
    extensions that want to call a callback stored on the environment.
    """
    fields = ('name',)
    name: str