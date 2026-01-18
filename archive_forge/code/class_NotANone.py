import textwrap
class NotANone(VarLibMergeError):
    """one of the values in a list was not empty when it should have been"""

    @property
    def offender(self):
        index = [x is not None for x in self.cause['got']].index(True)
        return (index, self._master_name(index))

    @property
    def details(self):
        cause, stack = (self.cause, self.stack)
        return f'{stack[0]}=={cause['got']}\n'