from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _DisplayMetadata(self, out, prepend, beneath_stub):
    """Prints the keys and values of the metadata for a node.

    Args:
      out: Output stream to which we print.
      prepend: String that precedes any information about this node to maintain
        a visible hierarchy.
      beneath_stub: String that preserves the indentation of the vertical lines.
    """
    if self.properties.metadata:
        additional_props = []
        for prop in self.properties.metadata.additionalProperties:
            additional_props.append('{}: {}'.format(prop.key, prop.value.string_value))
        metadata = '{}{} {}'.format(prepend, beneath_stub, ', '.join(sorted(additional_props)))
        out.Print(metadata)