from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
from . import types
from .array import ARRAY
from ...sql import coercions
from ...sql import elements
from ...sql import expression
from ...sql import functions
from ...sql import roles
from ...sql import schema
from ...sql.schema import ColumnCollectionConstraint
from ...sql.sqltypes import TEXT
from ...sql.visitors import InternalTraversal
class _regconfig_fn(functions.GenericFunction[_T]):
    inherit_cache = True

    def __init__(self, *args, **kwargs):
        args = list(args)
        if len(args) > 1:
            initial_arg = coercions.expect(roles.ExpressionElementRole, args.pop(0), name=getattr(self, 'name', None), apply_propagate_attrs=self, type_=types.REGCONFIG)
            initial_arg = [initial_arg]
        else:
            initial_arg = []
        addtl_args = [coercions.expect(roles.ExpressionElementRole, c, name=getattr(self, 'name', None), apply_propagate_attrs=self) for c in args]
        super().__init__(*initial_arg + addtl_args, **kwargs)