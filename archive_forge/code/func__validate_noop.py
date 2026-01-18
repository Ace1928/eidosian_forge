import gettext
import os
import re
import textwrap
import warnings
from . import declarative
def _validate_noop(self, value, state=None):
    """
        A validation method that doesn't do anything.
        """
    pass