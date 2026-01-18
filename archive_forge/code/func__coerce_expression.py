from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
def _coerce_expression(self, lambda_element, apply_propagate_attrs):
    """Run the tracker-generated expression through coercion rules.

        After the user-defined lambda has been invoked to produce a statement
        for re-use, run it through coercion rules to both check that it's the
        correct type of object and also to coerce it to its useful form.

        """
    parent_lambda = lambda_element.parent_lambda
    expr = self.expr
    if parent_lambda is None:
        if isinstance(expr, collections_abc.Sequence):
            self.expected_expr = [cast('ClauseElement', coercions.expect(lambda_element.role, sub_expr, apply_propagate_attrs=apply_propagate_attrs)) for sub_expr in expr]
            self.is_sequence = True
        else:
            self.expected_expr = cast('ClauseElement', coercions.expect(lambda_element.role, expr, apply_propagate_attrs=apply_propagate_attrs))
            self.is_sequence = False
    else:
        self.expected_expr = expr
        self.is_sequence = False
    if apply_propagate_attrs is not None:
        self.propagate_attrs = apply_propagate_attrs._propagate_attrs
    else:
        self.propagate_attrs = util.EMPTY_DICT