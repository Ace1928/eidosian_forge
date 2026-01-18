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
def PropertyValueFromString(type_, value_string, _auth_domain=None):
    """Returns an instance of a property value given a type and string value.

  The reverse of this method is just str() and type() of the python value.

  Note that this does *not* support non-UTC offsets in ISO 8601-formatted
  datetime strings, e.g. the -08:00 suffix in '2002-12-25 00:00:00-08:00'.
  It only supports -00:00 and +00:00 suffixes, which are UTC.

  Args:
    type_: A python class.
    value_string: A string representation of the value of the property.

  Returns:
    An instance of 'type'.

  Raises:
    ValueError if type_ is datetime and value_string has a timezone offset.
  """
    if type_ == datetime.datetime:
        value_string = value_string.strip()
        if value_string[-6] in ('+', '-'):
            if value_string[-5:] == '00:00':
                value_string = value_string[:-6]
            else:
                raise ValueError('Non-UTC offsets in datetimes are not supported.')
        split = value_string.split('.')
        iso_date = split[0]
        microseconds = 0
        if len(split) > 1:
            microseconds = int(split[1])
        time_struct = time.strptime(iso_date, '%Y-%m-%d %H:%M:%S')[0:6]
        value = datetime.datetime(*time_struct + (microseconds,))
        return value
    elif type_ == Rating:
        return Rating(int(value_string))
    elif type_ == bool:
        return value_string == 'True'
    elif type_ == users.User:
        return users.User(value_string, _auth_domain)
    elif type_ == type(None):
        return None
    return type_(value_string)