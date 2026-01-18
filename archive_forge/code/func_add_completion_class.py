import os
import re
import typing as t
from gettext import gettext as _
from .core import Argument
from .core import BaseCommand
from .core import Context
from .core import MultiCommand
from .core import Option
from .core import Parameter
from .core import ParameterSource
from .parser import split_arg_string
from .utils import echo
def add_completion_class(cls: ShellCompleteType, name: t.Optional[str]=None) -> ShellCompleteType:
    """Register a :class:`ShellComplete` subclass under the given name.
    The name will be provided by the completion instruction environment
    variable during completion.

    :param cls: The completion class that will handle completion for the
        shell.
    :param name: Name to register the class under. Defaults to the
        class's ``name`` attribute.
    """
    if name is None:
        name = cls.name
    _available_shells[name] = cls
    return cls