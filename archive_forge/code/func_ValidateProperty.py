from the atom and gd namespaces. For more information, see:
from __future__ import absolute_import
import base64
import calendar
import datetime
import os
import re
import time
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import namespace_manager
from googlecloudsdk.third_party.appengine.api import users
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
from googlecloudsdk.third_party.appengine.datastore import sortable_pb_encoder
from googlecloudsdk.third_party.appengine._internal import six_subset
def ValidateProperty(name, values, read_only=False):
    """Helper function for validating property values.

  Args:
    name: Name of the property this is for.
    value: Value for the property as a Python native type.
    read_only: deprecated

  Raises:
    BadPropertyError if the property name is invalid. BadValueError if the
    property did not validate correctly or the value was an empty list. Other
    exception types (like OverflowError) if the property value does not meet
    type-specific criteria.
  """
    ValidateString(name, 'property name', datastore_errors.BadPropertyError)
    values_type = type(values)
    if values_type is tuple:
        raise datastore_errors.BadValueError('May not use tuple property value; property %s is %s.' % (name, repr(values)))
    if values_type is not list:
        values = [values]
    try:
        for v in values:
            prop_validator = _VALIDATE_PROPERTY_VALUES.get(v.__class__)
            if prop_validator is None:
                raise datastore_errors.BadValueError('Unsupported type for property %s: %s' % (name, v.__class__))
            prop_validator(name, v)
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as msg:
        raise datastore_errors.BadValueError('Error type checking values for property %s: %s' % (name, msg))