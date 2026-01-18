from __future__ import annotations
import sys
from . import config
from . import exclusions
from .. import event
from .. import schema
from .. import types as sqltypes
from ..orm import mapped_column as _orm_mapped_column
from ..util import OrderedDict
class eq_type_affinity:
    """Helper to compare types inside of datastructures based on affinity.

    E.g.::

        eq_(
            inspect(connection).get_columns("foo"),
            [
                {
                    "name": "id",
                    "type": testing.eq_type_affinity(sqltypes.INTEGER),
                    "nullable": False,
                    "default": None,
                    "autoincrement": False,
                },
                {
                    "name": "data",
                    "type": testing.eq_type_affinity(sqltypes.NullType),
                    "nullable": True,
                    "default": None,
                    "autoincrement": False,
                },
            ],
        )

    """

    def __init__(self, target):
        self.target = sqltypes.to_instance(target)

    def __eq__(self, other):
        return self.target._type_affinity is other._type_affinity

    def __ne__(self, other):
        return self.target._type_affinity is not other._type_affinity