import enum
class Convert(Rule):
    """Indicates that this module should be converted."""

    def __str__(self):
        return 'Convert rule for {}'.format(self._prefix)

    def get_action(self, module):
        if self.matches(module.__name__):
            return Action.CONVERT
        return Action.NONE