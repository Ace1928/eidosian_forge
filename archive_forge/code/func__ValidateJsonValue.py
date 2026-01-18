import datetime
import json
import numbers
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def _ValidateJsonValue(json_value):
    entries = [(f, json_value.get_assigned_value(f.name)) for f in json_value.all_fields()]
    assigned_entries = [(f, value) for f, value in entries if value is not None]
    if len(assigned_entries) != 1:
        raise exceptions.InvalidDataError('Malformed JsonValue: %s' % json_value)