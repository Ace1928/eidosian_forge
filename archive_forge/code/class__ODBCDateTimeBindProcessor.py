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
class _ODBCDateTimeBindProcessor:
    """Add bind processors to handle datetimeoffset behaviors"""
    has_tz = False

    def bind_processor(self, dialect):

        def process(value):
            if value is None:
                return None
            elif isinstance(value, str):
                return value
            elif not value.tzinfo or (not self.timezone and (not self.has_tz)):
                return value
            else:
                dto_string = value.strftime('%Y-%m-%d %H:%M:%S.%f %z')
                dto_string = re.sub('([\\+\\-]\\d{2})([\\d\\.]+)$', '\\1:\\2', dto_string)
                return dto_string
        return process