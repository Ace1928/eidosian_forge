import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def _addvariant(self):
    """
        Create a more pretty-printable version of the assignment.
        """
    list_ = []
    for item in self.items():
        pair = (item[1], item[0])
        list_.append(pair)
    self.variant = list_
    return None