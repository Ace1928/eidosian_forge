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
class StatementError(SQLAlchemyError):
    """An error occurred during execution of a SQL statement.

    :class:`StatementError` wraps the exception raised
    during execution, and features :attr:`.statement`
    and :attr:`.params` attributes which supply context regarding
    the specifics of the statement which had an issue.

    The wrapped exception object is available in
    the :attr:`.orig` attribute.

    """
    statement: Optional[str] = None
    'The string SQL statement being invoked when this exception occurred.'
    params: Optional[_AnyExecuteParams] = None
    'The parameter list being used when this exception occurred.'
    orig: Optional[BaseException] = None
    'The original exception that was thrown.\n\n    '
    ismulti: Optional[bool] = None
    'multi parameter passed to repr_params().  None is meaningful.'
    connection_invalidated: bool = False

    def __init__(self, message: str, statement: Optional[str], params: Optional[_AnyExecuteParams], orig: Optional[BaseException], hide_parameters: bool=False, code: Optional[str]=None, ismulti: Optional[bool]=None):
        SQLAlchemyError.__init__(self, message, code=code)
        self.statement = statement
        self.params = params
        self.orig = orig
        self.ismulti = ismulti
        self.hide_parameters = hide_parameters
        self.detail: List[str] = []

    def add_detail(self, msg: str) -> None:
        self.detail.append(msg)

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (self.__class__, (self.args[0], self.statement, self.params, self.orig, self.hide_parameters, self.__dict__.get('code'), self.ismulti), {'detail': self.detail})

    @_preloaded.preload_module('sqlalchemy.sql.util')
    def _sql_message(self) -> str:
        util = _preloaded.sql_util
        details = [self._message()]
        if self.statement:
            stmt_detail = '[SQL: %s]' % self.statement
            details.append(stmt_detail)
            if self.params:
                if self.hide_parameters:
                    details.append('[SQL parameters hidden due to hide_parameters=True]')
                else:
                    params_repr = util._repr_params(self.params, 10, ismulti=self.ismulti)
                    details.append('[parameters: %r]' % params_repr)
        code_str = self._code_str()
        if code_str:
            details.append(code_str)
        return '\n'.join(['(%s)' % det for det in self.detail] + details)