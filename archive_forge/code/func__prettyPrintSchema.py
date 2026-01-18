from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
@util.positional(2)
def _prettyPrintSchema(self, schema, seen=None, dent=0):
    """Get pretty printed object prototype of schema.

    Args:
      schema: object, Parsed JSON schema.
      seen: list of string, Names of schema already seen. Used to handle
        recursive definitions.

    Returns:
      string, A string that contains a prototype object with
        comments that conforms to the given schema.
    """
    if seen is None:
        seen = []
    return _SchemaToStruct(schema, seen, dent=dent).to_str(self._prettyPrintByName)