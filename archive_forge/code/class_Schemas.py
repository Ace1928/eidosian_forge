from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
class Schemas(object):
    """Schemas for an API."""

    def __init__(self, discovery):
        """Constructor.

    Args:
      discovery: object, Deserialized discovery document from which we pull
        out the named schema.
    """
        self.schemas = discovery.get('schemas', {})
        self.pretty = {}

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

    def prettyPrintByName(self, name):
        """Get pretty printed object prototype from the schema name.

    Args:
      name: string, Name of schema in the discovery document.

    Returns:
      string, A string that contains a prototype object with
        comments that conforms to the given schema.
    """
        return self._prettyPrintByName(name, seen=[], dent=0)[:-2]

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

    def prettyPrintSchema(self, schema):
        """Get pretty printed object prototype of schema.

    Args:
      schema: object, Parsed JSON schema.

    Returns:
      string, A string that contains a prototype object with
        comments that conforms to the given schema.
    """
        return self._prettyPrintSchema(schema, dent=0)[:-2]

    def get(self, name, default=None):
        """Get deserialized JSON schema from the schema name.

    Args:
      name: string, Schema name.
      default: object, return value if name not found.
    """
        return self.schemas.get(name, default)