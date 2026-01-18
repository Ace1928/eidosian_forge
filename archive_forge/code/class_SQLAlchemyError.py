from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class SQLAlchemyError(HasDescriptionCode, Exception):
    """Generic error class."""

    def _message(self) -> str:
        text: str
        if len(self.args) == 1:
            arg_text = self.args[0]
            if isinstance(arg_text, bytes):
                text = compat.decode_backslashreplace(arg_text, 'utf-8')
            else:
                text = str(arg_text)
            return text
        else:
            return str(self.args)

    def _sql_message(self) -> str:
        message = self._message()
        if self.code:
            message = '%s %s' % (message, self._code_str())
        return message

    def __str__(self) -> str:
        return self._sql_message()