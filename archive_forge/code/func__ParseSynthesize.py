from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def _ParseSynthesize(self, args):
    """Parses the synthesize() transform args and returns a new transform.

    The args are a list of tuples. Each tuple is a schema that defines the
    synthesis of one resource list item. Each schema item is an attribute
    that defines the synthesis of one synthesized_resource attribute from
    an original_resource attribute.

    There are three kinds of attributes:

      name:literal
        The value for the name attribute in the synthesized resource is the
        literal value.
      name=key
        The value for the name attribute in the synthesized_resource is the
        value of key in the original_resource.
      key:
        All the attributes of the value of key in the original_resource are
        added to the attributes in the synthesized_resource.

    Args:
      args: The original synthesize transform args.

    Returns:
      A synthesize transform function that uses the schema from the parsed
      args.

    Example:
      This returns a list of two resource items:
        synthesize((name:up, upInfo), (name:down, downInfo))
      If upInfo and downInfo serialize to
        {"foo": 1, "bar": "yes"}
      and
        {"foo": 0, "bar": "no"}
      then the synthesized resource list is
        [{"name": "up", "foo": 1, "bar": "yes"},
        {"name": "down", "foo": 0, "bar": "no"}]
      which could be displayed by a nested table using
        synthesize(...):format="table(name, foo, bar)"
    """
    schemas = []
    for arg in args:
        lex = Lexer(arg)
        if not lex.IsCharacter('('):
            raise resource_exceptions.ExpressionSyntaxError('(...) args expected in synthesize() transform')
        schema = []
        for attr in lex.Args():
            if ':' in attr:
                name, literal = attr.split(':', 1)
                key = None
            elif '=' in attr:
                name, value = attr.split('=', 1)
                key = Lexer(value).Key()
                literal = None
            else:
                key = Lexer(attr).Key()
                name = None
                literal = None
            schema.append((name, key, literal))
        schemas.append(schema)

    def _Synthesize(r):
        """Synthesize a new resource list from the original resource r.

      Args:
        r: The original resource.

      Returns:
        The synthesized resource list.
      """
        synthesized_resource_list = []
        for schema in schemas:
            synthesized_resource = {}
            for attr in schema:
                name, key, literal = attr
                value = resource_property.Get(r, key, None) if key else literal
                if name:
                    synthesized_resource[name] = value
                elif isinstance(value, dict):
                    synthesized_resource.update(value)
            synthesized_resource_list.append(synthesized_resource)
        return synthesized_resource_list
    return _Synthesize