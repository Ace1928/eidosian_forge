from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
@util.positional(2)
def _prettyPrintByName(self, name, seen=None, dent=0):
    """Get pretty printed object prototype from the schema name.

    Args:
      name: string, Name of schema in the discovery document.
      seen: list of string, Names of schema already seen. Used to handle
        recursive definitions.

    Returns:
      string, A string that contains a prototype object with
        comments that conforms to the given schema.
    """
    if seen is None:
        seen = []
    if name in seen:
        return '# Object with schema name: %s' % name
    seen.append(name)
    if name not in self.pretty:
        self.pretty[name] = _SchemaToStruct(self.schemas[name], seen, dent=dent).to_str(self._prettyPrintByName)
    seen.pop()
    return self.pretty[name]