from typing import TYPE_CHECKING
from sqlalchemy import schema as sa_schema
from . import ops
from .base import Operations
from ..util.sqla_compat import _copy
from ..util.sqla_compat import sqla_14
def _count_constraint(constraint):
    return not isinstance(constraint, sa_schema.PrimaryKeyConstraint) and (not constraint._create_rule or constraint._create_rule(compiler))