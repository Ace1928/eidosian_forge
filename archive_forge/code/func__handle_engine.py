from __future__ import annotations
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
@staticmethod
def _handle_engine(engine: Engine | None=None) -> None:
    """
        Utility to assign the engine if it is provided.
        :param engine: A :class:`sqlalchemy.Engine` that defines the connection to a Postgres instance/cluster.
        """
    if engine:
        Entity._engine = engine