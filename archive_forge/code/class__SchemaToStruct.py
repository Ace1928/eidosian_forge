from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
class _SchemaToStruct(object):
    """Convert schema to a prototype object."""

    @util.positional(3)
    def __init__(self, schema, seen, dent=0):
        """Constructor.

    Args:
      schema: object, Parsed JSON schema.
      seen: list, List of names of schema already seen while parsing. Used to
        handle recursive definitions.
      dent: int, Initial indentation depth.
    """
        self.value = []
        self.string = None
        self.schema = schema
        self.dent = dent
        self.from_cache = None
        self.seen = seen

    def emit(self, text):
        """Add text as a line to the output.

    Args:
      text: string, Text to output.
    """
        self.value.extend(['  ' * self.dent, text, '\n'])

    def emitBegin(self, text):
        """Add text to the output, but with no line terminator.

    Args:
      text: string, Text to output.
      """
        self.value.extend(['  ' * self.dent, text])

    def emitEnd(self, text, comment):
        """Add text and comment to the output with line terminator.

    Args:
      text: string, Text to output.
      comment: string, Python comment.
    """
        if comment:
            divider = '\n' + '  ' * (self.dent + 2) + '# '
            lines = comment.splitlines()
            lines = [x.rstrip() for x in lines]
            comment = divider.join(lines)
            self.value.extend([text, ' # ', comment, '\n'])
        else:
            self.value.extend([text, '\n'])

    def indent(self):
        """Increase indentation level."""
        self.dent += 1

    def undent(self):
        """Decrease indentation level."""
        self.dent -= 1

    def _to_str_impl(self, schema):
        """Prototype object based on the schema, in Python code with comments.

    Args:
      schema: object, Parsed JSON schema file.

    Returns:
      Prototype object based on the schema, in Python code with comments.
    """
        stype = schema.get('type')
        if stype == 'object':
            self.emitEnd('{', schema.get('description', ''))
            self.indent()
            if 'properties' in schema:
                properties = schema.get('properties', {})
                sorted_properties = OrderedDict(sorted(properties.items()))
                for pname, pschema in six.iteritems(sorted_properties):
                    self.emitBegin('"%s": ' % pname)
                    self._to_str_impl(pschema)
            elif 'additionalProperties' in schema:
                self.emitBegin('"a_key": ')
                self._to_str_impl(schema['additionalProperties'])
            self.undent()
            self.emit('},')
        elif '$ref' in schema:
            schemaName = schema['$ref']
            description = schema.get('description', '')
            s = self.from_cache(schemaName, seen=self.seen)
            parts = s.splitlines()
            self.emitEnd(parts[0], description)
            for line in parts[1:]:
                self.emit(line.rstrip())
        elif stype == 'boolean':
            value = schema.get('default', 'True or False')
            self.emitEnd('%s,' % str(value), schema.get('description', ''))
        elif stype == 'string':
            value = schema.get('default', 'A String')
            self.emitEnd('"%s",' % str(value), schema.get('description', ''))
        elif stype == 'integer':
            value = schema.get('default', '42')
            self.emitEnd('%s,' % str(value), schema.get('description', ''))
        elif stype == 'number':
            value = schema.get('default', '3.14')
            self.emitEnd('%s,' % str(value), schema.get('description', ''))
        elif stype == 'null':
            self.emitEnd('None,', schema.get('description', ''))
        elif stype == 'any':
            self.emitEnd('"",', schema.get('description', ''))
        elif stype == 'array':
            self.emitEnd('[', schema.get('description'))
            self.indent()
            self.emitBegin('')
            self._to_str_impl(schema['items'])
            self.undent()
            self.emit('],')
        else:
            self.emit('Unknown type! %s' % stype)
            self.emitEnd('', '')
        self.string = ''.join(self.value)
        return self.string

    def to_str(self, from_cache):
        """Prototype object based on the schema, in Python code with comments.

    Args:
      from_cache: callable(name, seen), Callable that retrieves an object
         prototype for a schema with the given name. Seen is a list of schema
         names already seen as we recursively descend the schema definition.

    Returns:
      Prototype object based on the schema, in Python code with comments.
      The lines of the code will all be properly indented.
    """
        self.from_cache = from_cache
        return self._to_str_impl(self.schema)