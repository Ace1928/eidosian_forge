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
class DBAPIError(StatementError):
    """Raised when the execution of a database operation fails.

    Wraps exceptions raised by the DB-API underlying the
    database operation.  Driver-specific implementations of the standard
    DB-API exception types are wrapped by matching sub-types of SQLAlchemy's
    :class:`DBAPIError` when possible.  DB-API's ``Error`` type maps to
    :class:`DBAPIError` in SQLAlchemy, otherwise the names are identical.  Note
    that there is no guarantee that different DB-API implementations will
    raise the same exception type for any given error condition.

    :class:`DBAPIError` features :attr:`~.StatementError.statement`
    and :attr:`~.StatementError.params` attributes which supply context
    regarding the specifics of the statement which had an issue, for the
    typical case when the error was raised within the context of
    emitting a SQL statement.

    The wrapped exception object is available in the
    :attr:`~.StatementError.orig` attribute. Its type and properties are
    DB-API implementation specific.

    """
    code = 'dbapi'

    @overload
    @classmethod
    def instance(cls, statement: Optional[str], params: Optional[_AnyExecuteParams], orig: Exception, dbapi_base_err: Type[Exception], hide_parameters: bool=False, connection_invalidated: bool=False, dialect: Optional[Dialect]=None, ismulti: Optional[bool]=None) -> StatementError:
        ...

    @overload
    @classmethod
    def instance(cls, statement: Optional[str], params: Optional[_AnyExecuteParams], orig: DontWrapMixin, dbapi_base_err: Type[Exception], hide_parameters: bool=False, connection_invalidated: bool=False, dialect: Optional[Dialect]=None, ismulti: Optional[bool]=None) -> DontWrapMixin:
        ...

    @overload
    @classmethod
    def instance(cls, statement: Optional[str], params: Optional[_AnyExecuteParams], orig: BaseException, dbapi_base_err: Type[Exception], hide_parameters: bool=False, connection_invalidated: bool=False, dialect: Optional[Dialect]=None, ismulti: Optional[bool]=None) -> BaseException:
        ...

    @classmethod
    def instance(cls, statement: Optional[str], params: Optional[_AnyExecuteParams], orig: Union[BaseException, DontWrapMixin], dbapi_base_err: Type[Exception], hide_parameters: bool=False, connection_invalidated: bool=False, dialect: Optional[Dialect]=None, ismulti: Optional[bool]=None) -> Union[BaseException, DontWrapMixin]:
        if isinstance(orig, BaseException) and (not isinstance(orig, Exception)) or isinstance(orig, DontWrapMixin):
            return orig
        if orig is not None:
            if isinstance(orig, SQLAlchemyError) and statement:
                return StatementError('(%s.%s) %s' % (orig.__class__.__module__, orig.__class__.__name__, orig.args[0]), statement, params, orig, hide_parameters=hide_parameters, code=orig.code, ismulti=ismulti)
            elif not isinstance(orig, dbapi_base_err) and statement:
                return StatementError('(%s.%s) %s' % (orig.__class__.__module__, orig.__class__.__name__, orig), statement, params, orig, hide_parameters=hide_parameters, ismulti=ismulti)
            glob = globals()
            for super_ in orig.__class__.__mro__:
                name = super_.__name__
                if dialect:
                    name = dialect.dbapi_exception_translation_map.get(name, name)
                if name in glob and issubclass(glob[name], DBAPIError):
                    cls = glob[name]
                    break
        return cls(statement, params, orig, connection_invalidated=connection_invalidated, hide_parameters=hide_parameters, code=cls.code, ismulti=ismulti)

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (self.__class__, (self.statement, self.params, self.orig, self.hide_parameters, self.connection_invalidated, self.__dict__.get('code'), self.ismulti), {'detail': self.detail})

    def __init__(self, statement: Optional[str], params: Optional[_AnyExecuteParams], orig: BaseException, hide_parameters: bool=False, connection_invalidated: bool=False, code: Optional[str]=None, ismulti: Optional[bool]=None):
        try:
            text = str(orig)
        except Exception as e:
            text = 'Error in str() of DB-API-generated exception: ' + str(e)
        StatementError.__init__(self, '(%s.%s) %s' % (orig.__class__.__module__, orig.__class__.__name__, text), statement, params, orig, hide_parameters, code=code, ismulti=ismulti)
        self.connection_invalidated = connection_invalidated