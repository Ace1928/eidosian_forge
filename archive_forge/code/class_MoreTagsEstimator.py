import pytest
from sklearn.base import BaseEstimator
from sklearn.utils._tags import (
class MoreTagsEstimator:

    def _more_tags(self):
        return {'allow_nan': True}