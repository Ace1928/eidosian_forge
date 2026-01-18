import inspect
import os
import re
import string
import sys
import warnings
from functools import partial, wraps
class with_doc:
    """
    This decorator combines the docstrings of the provided and decorated objects
    to produce the final docstring for the decorated object.
    """

    def __init__(self, method, use_header=True):
        self.method = method
        if use_header:
            self.header = '\n\n    Notes\n    -----\n    '
        else:
            self.header = ''

    def __call__(self, new_method):
        new_doc = new_method.__doc__
        original_doc = self.method.__doc__
        header = self.header
        if original_doc and new_doc:
            new_method.__doc__ = '\n    {}\n    {}\n    {}\n        '.format(original_doc, header, new_doc)
        elif original_doc:
            new_method.__doc__ = original_doc
        return new_method