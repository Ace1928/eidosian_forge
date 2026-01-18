import pickle
import pytest
from sklearn.utils.deprecation import _is_deprecated, deprecated
@deprecated('qwerty')
class MockClass1:
    pass