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
def ValidateInteger(value, name='unused', exception=datastore_errors.BadValueError, empty_ok=False, zero_ok=False, negative_ok=False):
    """Raises an exception if value is not a valid integer.

  An integer is valid if it's not negative or empty and is an integer
  (either int or long).  The exception type raised can be specified
  with the exception argument; it defaults to BadValueError.

  Args:
    value: the value to validate.
    name: the name of this value; used in the exception message.
    exception: the type of exception to raise.
    empty_ok: allow None value.
    zero_ok: allow zero value.
    negative_ok: allow negative value.
  """
    if value is None and empty_ok:
        return
    if not isinstance(value, six_subset.integer_types):
        raise exception('%s should be an integer; received %s (a %s).' % (name, value, typename(value)))
    if not value and (not zero_ok):
        raise exception('%s must not be 0 (zero)' % name)
    if value < 0 and (not negative_ok):
        raise exception('%s must not be negative.' % name)