import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def _FormatCustomJsonMapping(mapping_type, mapping, descriptor):
    return '\n'.join(('encoding.AddCustomJson%sMapping(' % mapping_type, "    %s, '%s', '%s')" % (descriptor.full_name, mapping.python_name, mapping.json_name)))