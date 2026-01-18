from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
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