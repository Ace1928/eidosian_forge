from __future__ import annotations
import abc
import typing
from typing import Any
from typing import ClassVar
class IsolationLevelCharacteristic(ConnectionCharacteristic):
    transactional: ClassVar[bool] = True

    def reset_characteristic(self, dialect: Dialect, dbapi_conn: DBAPIConnection) -> None:
        dialect.reset_isolation_level(dbapi_conn)

    def set_characteristic(self, dialect: Dialect, dbapi_conn: DBAPIConnection, value: Any) -> None:
        dialect._assert_and_set_isolation_level(dbapi_conn, value)

    def get_characteristic(self, dialect: Dialect, dbapi_conn: DBAPIConnection) -> Any:
        return dialect.get_isolation_level(dbapi_conn)