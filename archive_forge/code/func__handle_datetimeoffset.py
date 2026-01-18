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
def _handle_datetimeoffset(dto_value):
    tup = struct.unpack('<6hI2h', dto_value)
    return datetime.datetime(tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6] // 1000, datetime.timezone(datetime.timedelta(hours=tup[7], minutes=tup[8])))