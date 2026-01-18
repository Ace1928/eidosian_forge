import textwrap
class VarLibMergeError(VarLibError):
    """Raised when input data cannot be merged into a variable font."""

    def __init__(self, merger=None, **kwargs):
        self.merger = merger
        if not kwargs:
            kwargs = {}
        if 'stack' in kwargs:
            self.stack = kwargs['stack']
            del kwargs['stack']
        else:
            self.stack = []
        self.cause = kwargs

    @property
    def reason(self):
        return self.__doc__

    def _master_name(self, ix):
        if self.merger is not None:
            ttf = self.merger.ttfs[ix]
            if 'name' in ttf and ttf['name'].getBestFullName():
                return ttf['name'].getBestFullName()
            elif hasattr(ttf.reader, 'file') and hasattr(ttf.reader.file, 'name'):
                return ttf.reader.file.name
        return f'master number {ix}'

    @property
    def offender(self):
        if 'expected' in self.cause and 'got' in self.cause:
            index = [x == self.cause['expected'] for x in self.cause['got']].index(False)
            master_name = self._master_name(index)
            if 'location' in self.cause:
                master_name = f'{master_name} ({self.cause['location']})'
            return (index, master_name)
        return (None, None)

    @property
    def details(self):
        if 'expected' in self.cause and 'got' in self.cause:
            offender_index, offender = self.offender
            got = self.cause['got'][offender_index]
            return f'Expected to see {self.stack[0]}=={self.cause['expected']!r}, instead saw {got!r}\n'
        return ''

    def __str__(self):
        offender_index, offender = self.offender
        location = ''
        if offender:
            location = f'\n\nThe problem is likely to be in {offender}:\n'
        context = ''.join(reversed(self.stack))
        basic = textwrap.fill(f"Couldn't merge the fonts, because {self.reason}. This happened while performing the following operation: {context}", width=78)
        return '\n\n' + basic + location + self.details