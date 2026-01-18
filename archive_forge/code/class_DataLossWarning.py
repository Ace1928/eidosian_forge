from __future__ import absolute_import, division, unicode_literals
import string
class DataLossWarning(UserWarning):
    """Raised when the current tree is unable to represent the input data"""
    pass