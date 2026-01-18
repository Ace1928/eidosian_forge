import platform
import six
from blessed.colorspace import CGA_COLORS, X11_COLORNAMES_TO_RGB
class FormattingOtherString(six.text_type):
    """
    A Unicode string which doubles as a callable for another sequence when called.

    This is used for the :meth:`~.Terminal.move_up`, ``down``, ``left``, and ``right()``
    family of functions::

        >>> from blessed import Terminal
        >>> term = Terminal()
        >>> move_right = FormattingOtherString(term.cuf1, term.cuf)
        >>> print(repr(move_right))
        u'\\x1b[C'
        >>> print(repr(move_right(666)))
        u'\\x1b[666C'
        >>> print(repr(move_right()))
        u'\\x1b[C'
    """

    def __new__(cls, direct, target):
        """
        Class constructor accepting 2 positional arguments.

        :arg str direct: capability name for direct formatting, eg ``('x' + term.right)``.
        :arg str target: capability name for callable, eg ``('x' + term.right(99))``.
        """
        new = six.text_type.__new__(cls, direct)
        new._callable = target
        return new

    def __getnewargs__(self):
        return (six.text_type.__new__(six.text_type, self), self._callable)

    def __call__(self, *args):
        """Return ``text`` by ``target``."""
        return self._callable(*args) if args else self