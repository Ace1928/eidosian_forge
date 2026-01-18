import textwrap
class UnsupportedFormat(VarLibMergeError):
    """an OpenType subtable (%s) had a format I didn't expect"""

    def __init__(self, merger=None, **kwargs):
        super().__init__(merger, **kwargs)
        if not self.stack:
            self.stack = ['.Format']

    @property
    def reason(self):
        s = self.__doc__ % self.cause['subtable']
        if 'value' in self.cause:
            s += f' ({self.cause['value']!r})'
        return s