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
def FromReferenceProperty(value):
    """Converts a reference PropertyValue to a Key.

  Args:
    value: entity_pb.PropertyValue

  Returns:
    Key

  Raises:
    BadValueError if the value is not a PropertyValue.
  """
    assert isinstance(value, entity_pb.PropertyValue)
    assert value.has_referencevalue()
    ref = value.referencevalue()
    key = Key()
    key_ref = key._Key__reference
    key_ref.set_app(ref.app())
    SetNamespace(key_ref, ref.name_space())
    for pathelem in ref.pathelement_list():
        key_ref.mutable_path().add_element().CopyFrom(pathelem)
    return key