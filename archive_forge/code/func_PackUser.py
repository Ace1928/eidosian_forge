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
def PackUser(name, value, pbvalue):
    """Packs a User property into a entity_pb.PropertyValue.

  Args:
    name: The name of the property as a string.
    value: A users.User instance.
    pbvalue: The entity_pb.PropertyValue to pack this value into.
  """
    pbvalue.mutable_uservalue().set_email(value.email().encode('utf-8'))
    pbvalue.mutable_uservalue().set_auth_domain(value.auth_domain().encode('utf-8'))
    pbvalue.mutable_uservalue().set_gaiaid(0)
    if value.user_id() is not None:
        pbvalue.mutable_uservalue().set_obfuscated_gaiaid(value.user_id().encode('utf-8'))
    if value.federated_identity() is not None:
        pbvalue.mutable_uservalue().set_federated_identity(value.federated_identity().encode('utf-8'))
    if value.federated_provider() is not None:
        pbvalue.mutable_uservalue().set_federated_provider(value.federated_provider().encode('utf-8'))