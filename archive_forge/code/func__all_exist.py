from __future__ import annotations
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
@classmethod
def _all_exist(cls, engine: Engine | None=None) -> bool:
    """
        Checks if all declared entities and grants exist in the cluster.
        :param engine:  A :class:`sqlalchemy.Engine` that defines the connection to a Postgres instance/cluster.
        """
    cls._handle_engine(engine)
    return cls._all_entities_exist() and cls._all_grants_exist()