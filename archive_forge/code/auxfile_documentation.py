from __future__ import unicode_literals, with_statement
import re
import pybtex.io
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
from pybtex import py3compat
Parse a file and return an AuxData object.