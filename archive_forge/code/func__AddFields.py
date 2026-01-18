from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import re
import textwrap
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
import six
def _AddFields(self, depth, parent, spec, fields):
    """Adds message fields to the YAML spec.

    Args:
      depth: The nested dict depth.
      parent: The parent spec (nested ordered dict to add fields to) of spec.
      spec: The nested ordered dict to add fields to.
      fields: A message spec fields dict to add to spec.
    """
    depth += 2
    for name, value in sorted(six.iteritems(fields)):
        description = value['description'].strip()
        if description.startswith(_OPTIONAL):
            description = description[len(_OPTIONAL):].strip()
        elif description.startswith(_REQUIRED):
            description = description[len(_REQUIRED):].strip()
        if description.startswith(_OUTPUT_ONLY):
            continue
        d = collections.OrderedDict()
        spec[name] = d
        d['description'] = _WrapDescription(depth, description)
        if value.get('repeated'):
            d['type'] = 'array'
            items = collections.OrderedDict(value.get('items', {}))
            d['items'] = items
            d = items
            depth += 2
        type_name = value.get('type', 'boolean')
        subfields = value.get('fields')
        if subfields:
            if name == 'additionalProperties':
                del spec[name]
                properties = collections.OrderedDict()
                self._AddFields(depth, d, properties, subfields)
                if properties:
                    parent[name] = properties
            else:
                d['$ref'] = self._GetSchemaFileName(type_name)
                self.Generate(type_name, subfields)
        elif type_name in self._generated:
            d['$ref'] = self._GetSchemaFileName(type_name)
        else:
            type_name = _NormalizeTypeName(type_name)
            if type_name == 'enum':
                enum = value.get('choices')
                d['type'] = 'string'
                d['enum'] = sorted([n for n, _ in six.iteritems(enum)])
            else:
                d['type'] = type_name