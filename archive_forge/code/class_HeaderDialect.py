import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
class HeaderDialect(csv.Dialect):
    """CSV dialect to use for the "header" option data."""
    delimiter = ','
    quotechar = '"'
    escapechar = '\\'
    doublequote = False
    skipinitialspace = True
    strict = True
    lineterminator = '\n'
    quoting = csv.QUOTE_MINIMAL