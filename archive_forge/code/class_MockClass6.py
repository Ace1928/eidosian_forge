import pickle
import pytest
from sklearn.utils.deprecation import _is_deprecated, deprecated
@deprecated('a message')
class MockClass6:
    """A deprecated class that overrides __new__."""

    def __new__(cls, *args, **kwargs):
        assert len(args) > 0
        return super().__new__(cls)