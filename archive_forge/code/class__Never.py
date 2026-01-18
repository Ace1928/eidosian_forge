from ._impl import Mismatch
class _Never:
    """Never matches."""

    def __str__(self):
        return 'Never()'

    def match(self, value):
        return Mismatch(f'Inevitable mismatch on {value!r}')