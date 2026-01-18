from winappdbg.textio import HexInput
from winappdbg.util import StaticClass, MemoryAddresses
from winappdbg import win32
import warnings
class RegExpPattern(Pattern):
    """
    Regular expression pattern.

    @type pattern: str
    @ivar pattern: Regular expression in text form.

    @type flags: int
    @ivar flags: Regular expression flags.

    @type regexp: re.compile
    @ivar regexp: Regular expression in compiled form.

    @type maxLength: int
    @ivar maxLength:
        Maximum expected length of the strings matched by this regular
        expression.

        This value will be used to calculate the required buffer size when
        doing buffered searches.

        Ideally it should be an exact value, but in some cases it's not
        possible to calculate so an upper limit should be given instead.

        If that's not possible either, C{None} should be used. That will
        cause an exception to be raised if this pattern is used in a
        buffered search.
    """

    def __init__(self, regexp, flags=0, maxLength=None):
        """
        @type  regexp: str
        @param regexp: Regular expression string.

        @type  flags: int
        @param flags: Regular expression flags.

        @type  maxLength: int
        @param maxLength: Maximum expected length of the strings matched by
            this regular expression.

            This value will be used to calculate the required buffer size when
            doing buffered searches.

            Ideally it should be an exact value, but in some cases it's not
            possible to calculate so an upper limit should be given instead.

            If that's not possible either, C{None} should be used. That will
            cause an exception to be raised if this pattern is used in a
            buffered search.
        """
        self.pattern = regexp
        self.flags = flags
        self.regexp = re.compile(regexp, flags)
        self.maxLength = maxLength

    def __len__(self):
        """
        Returns the maximum expected length of the strings matched by this
        pattern. This value is taken from the C{maxLength} argument of the
        constructor if this class.

        Ideally it should be an exact value, but in some cases it's not
        possible to calculate so an upper limit should be returned instead.

        If that's not possible either an exception must be raised.

        This value will be used to calculate the required buffer size when
        doing buffered searches.
        """
        if self.maxLength is None:
            raise NotImplementedError()
        return self.maxLength

    def find(self, buffer, pos=None):
        if not pos:
            pos = 0
        match = self.regexp.search(buffer, pos)
        if match:
            start, end = match.span()
            return (start, end - start)
        return (-1, 0)