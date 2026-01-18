import os
import platform
import sys
import subprocess
import re
import warnings
from Bio import BiopythonDeprecationWarning
class _Argument(_AbstractParameter):
    """Represent an argument on a commandline.

    The names argument should be a list containing one string.
    This must be a valid Python identifier as it is used as the
    property name and as a keyword argument, and should therefore
    follow PEP8 naming.
    """

    def __init__(self, names, description, filename=False, checker_function=None, is_required=False):
        self.names = names
        if not isinstance(description, str):
            raise TypeError(f'Should be a string: {description!r} for {names[-1]}')
        self.is_filename = filename
        self.checker_function = checker_function
        self.description = description
        self.is_required = is_required
        self.is_set = False
        self.value = None

    def __str__(self):
        if self.value is None:
            return ' '
        elif self.is_filename:
            return f'{_escape_filename(self.value)} '
        else:
            return f'{self.value} '