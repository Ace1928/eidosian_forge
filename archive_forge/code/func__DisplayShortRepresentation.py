from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _DisplayShortRepresentation(self, out, prepend, beneath_stub):
    if self.properties.shortRepresentation:
        short_rep = '{}{} {}'.format(prepend, beneath_stub, self.properties.shortRepresentation.description)
        out.Print(short_rep)