import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _FetchRemapping(type_name, mapping_type, python_name=None, json_name=None, mappings=None):
    """Common code for fetching a key or value from a remapping dict."""
    if python_name and json_name:
        raise exceptions.InvalidDataError('Cannot specify both python_name and json_name for %s remapping' % mapping_type)
    if not (python_name or json_name):
        raise exceptions.InvalidDataError('Must specify either python_name or json_name for %s remapping' % (mapping_type,))
    field_remappings = mappings.get(type_name, {})
    if field_remappings:
        if python_name:
            return field_remappings.get(python_name)
        elif json_name:
            if json_name in list(field_remappings.values()):
                return [k for k in field_remappings if field_remappings[k] == json_name][0]
    return None