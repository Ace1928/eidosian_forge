import enum
class DoNotConvert(Rule):
    """Indicates that this module should be not converted."""

    def __str__(self):
        return 'DoNotConvert rule for {}'.format(self._prefix)

    def get_action(self, module):
        if self.matches(module.__name__):
            return Action.DO_NOT_CONVERT
        return Action.NONE