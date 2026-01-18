from ._impl import Mismatch
class _Always:
    """Always matches."""

    def __str__(self):
        return 'Always()'

    def match(self, value):
        return None