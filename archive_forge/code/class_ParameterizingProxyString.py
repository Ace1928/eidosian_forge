import platform
import six
from blessed.colorspace import CGA_COLORS, X11_COLORNAMES_TO_RGB
class ParameterizingProxyString(six.text_type):
    """
    A Unicode string which can be called to proxy missing termcap entries.

    This class supports the function :func:`get_proxy_string`, and mirrors
    the behavior of :class:`ParameterizingString`, except that instead of
    a capability name, receives a format string, and callable to filter the
    given positional ``*args`` of :meth:`ParameterizingProxyString.__call__`
    into a terminal sequence.

    For example::

        >>> from blessed import Terminal
        >>> term = Terminal('screen')
        >>> hpa = ParameterizingString(term.hpa, term.normal, 'hpa')
        >>> hpa(9)
        u''
        >>> fmt = u'\\x1b[{0}G'
        >>> fmt_arg = lambda *arg: (arg[0] + 1,)
        >>> hpa = ParameterizingProxyString((fmt, fmt_arg), term.normal, 'hpa')
        >>> hpa(9)
        u'\\x1b[10G'
    """

    def __new__(cls, fmt_pair, normal=u'', name=u'<not specified>'):
        """
        Class constructor accepting 4 positional arguments.

        :arg tuple fmt_pair: Two element tuple containing:
            - format string suitable for displaying terminal sequences
            - callable suitable for receiving  __call__ arguments for formatting string
        :arg str normal: terminating sequence for this capability (optional).
        :arg str name: name of this terminal capability (optional).
        """
        assert isinstance(fmt_pair, tuple), fmt_pair
        assert callable(fmt_pair[1]), fmt_pair[1]
        new = six.text_type.__new__(cls, fmt_pair[0])
        new._fmt_args = fmt_pair[1]
        new._normal = normal
        new._name = name
        return new

    def __call__(self, *args):
        """
        Returning :class:`FormattingString` instance for given parameters.

        Arguments are determined by the capability.  For example, ``hpa``
        (move_x) receives only a single integer, whereas ``cup`` (move)
        receives two integers.  See documentation in terminfo(5) for the
        given capability.

        :rtype: FormattingString
        :returns: Callable string for given parameters
        """
        return FormattingString(self.format(*self._fmt_args(*args)), self._normal)