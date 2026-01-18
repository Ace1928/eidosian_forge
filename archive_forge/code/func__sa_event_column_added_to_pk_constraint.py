from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from .base import SchemaEventTarget
from .. import event
def _sa_event_column_added_to_pk_constraint(self, const: Constraint, col: Column[Any]) -> None:
    """internal event hook used for primary key naming convention
        updates.

        """