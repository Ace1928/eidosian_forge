from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import encoding
import six
class ValuePrinter(CsvPrinter):
    """A printer for printing value data.

  CSV with no heading and <TAB> separator instead of <COMMA>. Used to retrieve
  individual resource values. This format requires a projection to define the
  value(s) to be printed.

  To use *\\n* or *\\t* as an attribute value please escape the *\\* with your
  shell's escape sequence, example *separator="\\\\n"* for bash.

  Printer attributes:
    delimiter="string": The string printed between list value items,
      default ";".
    quote: "..." quote values that contain delimiter, separator or terminator
      strings.
    separator="string": The string printed between values, default
      "\\t" (tab).
    terminator="string": The string printed after each record, default
      "\\n" (newline).
  """

    def __init__(self, *args, **kwargs):
        super(ValuePrinter, self).__init__(*args, **kwargs)
        self._heading_printed = True
        self._delimiter = self.attributes.get('delimiter', ';')
        self._quote = '"' if self.attributes.get('quote', 0) else None
        self._separator = self.attributes.get('separator', '\t')
        self._terminator = self.attributes.get('terminator', '\n')