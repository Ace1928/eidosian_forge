import operator
import types
import typing as t
from _string import formatter_field_name_split  # type: ignore
from collections import abc
from collections import deque
from string import Formatter
from markupsafe import EscapeFormatter
from markupsafe import Markup
from .environment import Environment
from .exceptions import SecurityError
from .runtime import Context
from .runtime import Undefined
class SandboxedEnvironment(Environment):
    """The sandboxed environment.  It works like the regular environment but
    tells the compiler to generate sandboxed code.  Additionally subclasses of
    this environment may override the methods that tell the runtime what
    attributes or functions are safe to access.

    If the template tries to access insecure code a :exc:`SecurityError` is
    raised.  However also other exceptions may occur during the rendering so
    the caller has to ensure that all exceptions are caught.
    """
    sandboxed = True
    default_binop_table: t.Dict[str, t.Callable[[t.Any, t.Any], t.Any]] = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv, '//': operator.floordiv, '**': operator.pow, '%': operator.mod}
    default_unop_table: t.Dict[str, t.Callable[[t.Any], t.Any]] = {'+': operator.pos, '-': operator.neg}
    intercepted_binops: t.FrozenSet[str] = frozenset()
    intercepted_unops: t.FrozenSet[str] = frozenset()

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.globals['range'] = safe_range
        self.binop_table = self.default_binop_table.copy()
        self.unop_table = self.default_unop_table.copy()

    def is_safe_attribute(self, obj: t.Any, attr: str, value: t.Any) -> bool:
        """The sandboxed environment will call this method to check if the
        attribute of an object is safe to access.  Per default all attributes
        starting with an underscore are considered private as well as the
        special attributes of internal python objects as returned by the
        :func:`is_internal_attribute` function.
        """
        return not (attr.startswith('_') or is_internal_attribute(obj, attr))

    def is_safe_callable(self, obj: t.Any) -> bool:
        """Check if an object is safely callable. By default callables
        are considered safe unless decorated with :func:`unsafe`.

        This also recognizes the Django convention of setting
        ``func.alters_data = True``.
        """
        return not (getattr(obj, 'unsafe_callable', False) or getattr(obj, 'alters_data', False))

    def call_binop(self, context: Context, operator: str, left: t.Any, right: t.Any) -> t.Any:
        """For intercepted binary operator calls (:meth:`intercepted_binops`)
        this function is executed instead of the builtin operator.  This can
        be used to fine tune the behavior of certain operators.

        .. versionadded:: 2.6
        """
        return self.binop_table[operator](left, right)

    def call_unop(self, context: Context, operator: str, arg: t.Any) -> t.Any:
        """For intercepted unary operator calls (:meth:`intercepted_unops`)
        this function is executed instead of the builtin operator.  This can
        be used to fine tune the behavior of certain operators.

        .. versionadded:: 2.6
        """
        return self.unop_table[operator](arg)

    def getitem(self, obj: t.Any, argument: t.Union[str, t.Any]) -> t.Union[t.Any, Undefined]:
        """Subscribe an object from sandboxed code."""
        try:
            return obj[argument]
        except (TypeError, LookupError):
            if isinstance(argument, str):
                try:
                    attr = str(argument)
                except Exception:
                    pass
                else:
                    try:
                        value = getattr(obj, attr)
                    except AttributeError:
                        pass
                    else:
                        if self.is_safe_attribute(obj, argument, value):
                            return value
                        return self.unsafe_undefined(obj, argument)
        return self.undefined(obj=obj, name=argument)

    def getattr(self, obj: t.Any, attribute: str) -> t.Union[t.Any, Undefined]:
        """Subscribe an object from sandboxed code and prefer the
        attribute.  The attribute passed *must* be a bytestring.
        """
        try:
            value = getattr(obj, attribute)
        except AttributeError:
            try:
                return obj[attribute]
            except (TypeError, LookupError):
                pass
        else:
            if self.is_safe_attribute(obj, attribute, value):
                return value
            return self.unsafe_undefined(obj, attribute)
        return self.undefined(obj=obj, name=attribute)

    def unsafe_undefined(self, obj: t.Any, attribute: str) -> Undefined:
        """Return an undefined object for unsafe attributes."""
        return self.undefined(f'access to attribute {attribute!r} of {type(obj).__name__!r} object is unsafe.', name=attribute, obj=obj, exc=SecurityError)

    def format_string(self, s: str, args: t.Tuple[t.Any, ...], kwargs: t.Dict[str, t.Any], format_func: t.Optional[t.Callable]=None) -> str:
        """If a format call is detected, then this is routed through this
        method so that our safety sandbox can be used for it.
        """
        formatter: SandboxedFormatter
        if isinstance(s, Markup):
            formatter = SandboxedEscapeFormatter(self, escape=s.escape)
        else:
            formatter = SandboxedFormatter(self)
        if format_func is not None and format_func.__name__ == 'format_map':
            if len(args) != 1 or kwargs:
                raise TypeError(f'format_map() takes exactly one argument {len(args) + (kwargs is not None)} given')
            kwargs = args[0]
            args = ()
        rv = formatter.vformat(s, args, kwargs)
        return type(s)(rv)

    def call(__self, __context: Context, __obj: t.Any, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Call an object from sandboxed code."""
        fmt = inspect_format_method(__obj)
        if fmt is not None:
            return __self.format_string(fmt, args, kwargs, __obj)
        if not __self.is_safe_callable(__obj):
            raise SecurityError(f'{__obj!r} is not safely callable')
        return __context.call(__obj, *args, **kwargs)