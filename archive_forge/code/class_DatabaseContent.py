from __future__ import annotations
from typing import Sequence, Type
from lazyops.libs.dbinit.base import Inspector, TextClause, inspect, text, DeclarativeBase
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.database import Database
from lazyops.libs.dbinit.entities.database_entity import DatabaseEntity
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
from lazyops.libs.dbinit.entities.schema import Schema
from lazyops.libs.dbinit.mixins.grantable import Grantable
from lazyops.libs.dbinit.mixins.sql import SQLBase
class DatabaseContent(DatabaseEntity):
    """
    Represents a `sqlalchemy.orm.DeclarativeBase` and other helpful functionality.
    """

    def __init__(self, name: str, sqlalchemy_base: Type[DeclarativeBase], database: Database, schemas: Sequence[Schema] | None=None, depends_on: Sequence[Entity] | None=None, check_if_exists: bool | None=None):
        """
        :param name: Unique name of the entity. Must be unique within a database.
        :param sqlalchemy_base: The `sqlalchemy.orm.DeclarativeBase` to refer to a collection of tables.
        :param schemas: Any schemas that need to be created prior to the tables specified in the sqlalchemy_base.
        :param database: The :class:`lazyops.libs.dbinit.entities.Database` that this entity belongs to.
        :param depends_on: Any entities that should be created before this one.
        :param check_if_exists: Flag to set existence check behavior. If `True`, will raise an exception during _safe_create if the entity already exists, and will raise an exception during _safe_drop if the entity does not exist.
        """
        super().__init__(name=name, depends_on=depends_on, database=database, check_if_exists=check_if_exists)
        self.base = sqlalchemy_base
        self.schemas = schemas
        self.tables: dict[str, Table] = {table.name: Table(name=table.name, database_content=self, schema=table.schema) for table in self.base.metadata.tables.values()}

    def _create(self) -> None:
        self.base.metadata.create_all(self.database.db_engine())

    def _exists(self) -> bool:
        inspector: Inspector = inspect(self.database.db_engine())
        return all((inspector.has_table(table_name=table.name, schema=table.schema) for table in self.base.metadata.tables.values()))

    def _drop(self) -> None:
        self.base.metadata.drop_all(self.database.db_engine())