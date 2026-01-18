import datetime
import decimal
import re
import struct
from .base import _MSDateTime
from .base import _MSUnicode
from .base import _MSUnicodeText
from .base import BINARY
from .base import DATETIMEOFFSET
from .base import MSDialect
from .base import MSExecutionContext
from .base import VARBINARY
from .json import JSON as _MSJson
from .json import JSONIndexType as _MSJsonIndexType
from .json import JSONPathType as _MSJsonPathType
from ... import exc
from ... import types as sqltypes
from ... import util
from ...connectors.pyodbc import PyODBCConnector
from ...engine import cursor as _cursor
class MSDialect_pyodbc(PyODBCConnector, MSDialect):
    supports_statement_cache = True
    supports_sane_rowcount_returning = False
    execution_ctx_cls = MSExecutionContext_pyodbc
    colspecs = util.update_copy(MSDialect.colspecs, {sqltypes.Numeric: _MSNumeric_pyodbc, sqltypes.Float: _MSFloat_pyodbc, BINARY: _BINARY_pyodbc, sqltypes.DateTime: _ODBCDateTime, DATETIMEOFFSET: _ODBCDATETIMEOFFSET, VARBINARY: _VARBINARY_pyodbc, sqltypes.VARBINARY: _VARBINARY_pyodbc, sqltypes.LargeBinary: _VARBINARY_pyodbc, sqltypes.String: _String_pyodbc, sqltypes.Unicode: _Unicode_pyodbc, sqltypes.UnicodeText: _UnicodeText_pyodbc, sqltypes.JSON: _JSON_pyodbc, sqltypes.JSON.JSONIndexType: _JSONIndexType_pyodbc, sqltypes.JSON.JSONPathType: _JSONPathType_pyodbc, sqltypes.Enum: sqltypes.Enum})

    def __init__(self, fast_executemany=False, use_setinputsizes=True, **params):
        super().__init__(use_setinputsizes=use_setinputsizes, **params)
        self.use_scope_identity = self.use_scope_identity and self.dbapi and hasattr(self.dbapi.Cursor, 'nextset')
        self._need_decimal_fix = self.dbapi and self._dbapi_version() < (2, 1, 8)
        self.fast_executemany = fast_executemany
        if fast_executemany:
            self.use_insertmanyvalues_wo_returning = False

    def _get_server_version_info(self, connection):
        try:
            raw = connection.exec_driver_sql("SELECT CAST(SERVERPROPERTY('ProductVersion') AS VARCHAR)").scalar()
        except exc.DBAPIError:
            return super()._get_server_version_info(connection)
        else:
            version = []
            r = re.compile('[.\\-]')
            for n in r.split(raw):
                try:
                    version.append(int(n))
                except ValueError:
                    pass
            return tuple(version)

    def on_connect(self):
        super_ = super().on_connect()

        def on_connect(conn):
            if super_ is not None:
                super_(conn)
            self._setup_timestampoffset_type(conn)
        return on_connect

    def _setup_timestampoffset_type(self, connection):

        def _handle_datetimeoffset(dto_value):
            tup = struct.unpack('<6hI2h', dto_value)
            return datetime.datetime(tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6] // 1000, datetime.timezone(datetime.timedelta(hours=tup[7], minutes=tup[8])))
        odbc_SQL_SS_TIMESTAMPOFFSET = -155
        connection.add_output_converter(odbc_SQL_SS_TIMESTAMPOFFSET, _handle_datetimeoffset)

    def do_executemany(self, cursor, statement, parameters, context=None):
        if self.fast_executemany:
            cursor.fast_executemany = True
        super().do_executemany(cursor, statement, parameters, context=context)

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.Error):
            code = e.args[0]
            if code in {'08S01', '01000', '01002', '08003', '08007', '08S02', '08001', 'HYT00', 'HY010', '10054'}:
                return True
        return super().is_disconnect(e, connection, cursor)