from __future__ import annotations
from typing import Sequence
from lazyops.libs.dbinit.base import Engine, TextClause, create_engine, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.cluster_entity import ClusterEntity
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
from lazyops.libs.dbinit.mixins.grantable import Grantable
def _grants_exist_statement(self) -> TextClause:
    """
        The SQL statement that checks to see what grants exist.
        :return: A single :class:`sqlalchemy.TextClause` containing the SQL to check what grants exist on this entity.
        """
    return text('SELECT unnest(datacl) AS acl FROM pg_catalog.pg_database WHERE datname=:db_name').bindparams(db_name=self.name)