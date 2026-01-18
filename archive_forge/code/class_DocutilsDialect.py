import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
class DocutilsDialect(csv.Dialect):
    """CSV dialect for `csv_table` directive."""
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = True
    strict = True
    lineterminator = '\n'
    quoting = csv.QUOTE_MINIMAL

    def __init__(self, options):
        if 'delim' in options:
            self.delimiter = CSVTable.encode_for_csv(options['delim'])
        if 'keepspace' in options:
            self.skipinitialspace = False
        if 'quote' in options:
            self.quotechar = CSVTable.encode_for_csv(options['quote'])
        if 'escape' in options:
            self.doublequote = False
            self.escapechar = CSVTable.encode_for_csv(options['escape'])
        csv.Dialect.__init__(self)