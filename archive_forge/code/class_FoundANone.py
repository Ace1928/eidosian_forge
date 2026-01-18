import textwrap
class FoundANone(VarLibMergeError):
    """one of the values in a list was empty when it shouldn't have been"""

    @property
    def offender(self):
        index = [x is None for x in self.cause['got']].index(True)
        return (index, self._master_name(index))

    @property
    def details(self):
        cause, stack = (self.cause, self.stack)
        return f'{stack[0]}=={cause['got']}\n'