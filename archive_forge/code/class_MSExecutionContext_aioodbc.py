from __future__ import annotations
from .pyodbc import MSDialect_pyodbc
from .pyodbc import MSExecutionContext_pyodbc
from ...connectors.aioodbc import aiodbcConnector
class MSExecutionContext_aioodbc(MSExecutionContext_pyodbc):

    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(server_side=True)