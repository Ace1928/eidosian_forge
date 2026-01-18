from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
import six
def _StringQuote(s, quote='"', escape='\\'):
    """Returns <quote>s<quote> with <escape> and <quote> in s escaped.

  s.encode('string-escape') does not work with type(s) == unicode.

  Args:
    s: The string to quote.
    quote: The outer quote character.
    escape: The enclosed escape character.

  Returns:
    <quote>s<quote> with <escape> and <quote> in s escaped.
  """
    entity = {'\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
    chars = []
    if quote:
        chars.append(quote)
    for c in s:
        if c in (escape, quote):
            chars.append(escape)
        elif c in entity:
            c = entity[c]
        chars.append(c)
    if quote:
        chars.append(quote)
    return ''.join(chars)