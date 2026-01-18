from __future__ import annotations
from argparse import Namespace
import collections
import inspect
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from . import mock
from . import requirements as _requirements
from .util import fail
from .. import util
class _NullFixtureFunctions:

    def _null_decorator(self):

        def go(fn):
            return fn
        return go

    def skip_test_exception(self, *arg, **kw):
        return Exception()

    @property
    def add_to_marker(self):
        return mock.Mock()

    def mark_base_test_class(self):
        return self._null_decorator()

    def combinations(self, *arg_sets, **kw):
        return self._null_decorator()

    def param_ident(self, *parameters):
        return self._null_decorator()

    def fixture(self, *arg, **kw):
        return self._null_decorator()

    def get_current_test_name(self):
        return None

    def async_test(self, fn):
        return fn