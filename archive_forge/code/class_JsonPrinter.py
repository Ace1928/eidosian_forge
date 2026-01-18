from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core.resource import resource_printer_base
import six
class JsonPrinter(resource_printer_base.ResourcePrinter):
    """Prints resource records as a JSON list.

  [JSON](http://www.json.org), JavaScript Object Notation.

  Printer attributes:
    no-undefined: Does not display resource data items with null values.

  Attributes:
    _buffer: Buffer stream for record item indentation.
    _delimiter: Delimiter string before the next record.
    _empty: True if no records were output.
    _indent: Resource item indentation.
  """
    _BEGIN_DELIMITER = '[\n'

    def __init__(self, *args, **kwargs):
        super(JsonPrinter, self).__init__(*args, retain_none_values=True, **kwargs)
        self._empty = True
        self._delimiter = self._BEGIN_DELIMITER
        self._indent = ' ' * resource_printer_base.STRUCTURED_INDENTATION

    def __Dump(self, resource):
        data = json.dumps(resource, ensure_ascii=False, indent=resource_printer_base.STRUCTURED_INDENTATION, separators=(',', ': '), sort_keys=True)
        return six.text_type(data)

    def _AddRecord(self, record, delimit=True):
        """Prints one element of a JSON-serializable Python object resource list.

    Allows intermingled delimit=True and delimit=False.

    Args:
      record: A JSON-serializable object.
      delimit: Dump one record if False, used by PrintSingleRecord().
    """
        self._empty = False
        output = self.__Dump(record)
        if delimit:
            delimiter = self._delimiter + self._indent
            self._delimiter = ',\n'
            for line in output.split('\n'):
                self._out.write(delimiter + line)
                delimiter = '\n' + self._indent
        else:
            if self._delimiter != self._BEGIN_DELIMITER:
                self._out.write('\n]\n')
                self._delimiter = self._BEGIN_DELIMITER
            self._out.write(output)
            self._out.write('\n')

    def Finish(self):
        """Prints the final delimiter and preps for the next resource list."""
        if self._empty:
            self._out.write('[]\n')
        elif self._delimiter != self._BEGIN_DELIMITER:
            self._out.write('\n]\n')
            self._delimiter = self._BEGIN_DELIMITER
        super(JsonPrinter, self).Finish()