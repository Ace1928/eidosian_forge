from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _DisplayKindAndName(self, out, prepend, stub):
    """Prints the kind of the node (SCALAR or RELATIONAL) and its name."""
    kind_and_name = '{}{} {} {}'.format(prepend, stub, self.properties.kind, self.properties.displayName)
    out.Print(kind_and_name)