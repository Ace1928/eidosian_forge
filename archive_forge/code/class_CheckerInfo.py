import abc
from collections import namedtuple
from datetime import datetime
import pyrfc3339
from ._caveat import parse_caveat
from ._conditions import (
from ._declared import DECLARED_KEY
from ._namespace import Namespace
from ._operation import OP_KEY
from ._time import TIME_KEY
from ._utils import condition_with_prefix
class CheckerInfo(namedtuple('CheckInfo', 'prefix name ns check')):
    """CheckerInfo holds information on a registered checker.
    """
    __slots__ = ()

    def __new__(cls, prefix, name, ns, check=None):
        """
        :param check holds the actual checker function which takes an auth
        context and a condition and arg string as arguments.
        :param prefix holds the prefix for the checker condition as string.
        :param name holds the name of the checker condition as string.
        :param ns holds the namespace URI for the checker's schema as
        Namespace.
        """
        return super(CheckerInfo, cls).__new__(cls, prefix, name, ns, check)