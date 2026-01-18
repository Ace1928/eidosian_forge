import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re

        Compares a dictionary of ``path: content`` to the
        found files.  Comparison is done by equality, or the
        ``comparison(actual_content, expected_content)`` function given.

        Returns dictionary of differences, keyed by path.  Each
        difference is either noted, or the output of
        ``differ(actual_content, expected_content)`` is given.

        If a file does not exist and ``not_found`` is given, then
        ``not_found(path)`` is put in.
        