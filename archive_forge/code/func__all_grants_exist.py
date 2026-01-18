from __future__ import annotations
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
@classmethod
def _all_grants_exist(cls, engine: Engine | None=None) -> bool:
    """
        Checks if all declared grants exist in the cluster.
        :param engine:  A :class:`sqlalchemy.Engine` that defines the connection to a Postgres instance/cluster.
        """
    cls._handle_engine(engine)
    return all((role._grants_exist() for role in Entity.entities if isinstance(role, Role)))