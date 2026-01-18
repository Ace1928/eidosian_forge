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
def PropertyValueToKeyValue(prop_value):
    """Converts a entity_pb.PropertyValue into a comparable hashable "key" value.

  The values produces by this function mimic the native ording of the datastore
  and uniquely identify the given PropertyValue.

  Args:
    prop_value: The entity_pb.PropertyValue from which to construct the
      key value.

  Returns:
    A comparable and hashable representation of the given property value.
  """
    if not isinstance(prop_value, entity_pb.PropertyValue):
        raise datastore_errors.BadArgumentError('prop_value arg expected to be entity_pb.PropertyValue (%r)' % (prop_value,))
    if prop_value.has_stringvalue():
        return (entity_pb.PropertyValue.kstringValue, prop_value.stringvalue())
    if prop_value.has_int64value():
        return (entity_pb.PropertyValue.kint64Value, prop_value.int64value())
    if prop_value.has_booleanvalue():
        return (entity_pb.PropertyValue.kbooleanValue, prop_value.booleanvalue())
    if prop_value.has_doublevalue():
        encoder = sortable_pb_encoder.Encoder()
        encoder.putDouble(prop_value.doublevalue())
        return (entity_pb.PropertyValue.kdoubleValue, tuple(encoder.buf))
    if prop_value.has_pointvalue():
        return (entity_pb.PropertyValue.kPointValueGroup, prop_value.pointvalue().x(), prop_value.pointvalue().y())
    if prop_value.has_referencevalue():
        return ReferenceToKeyValue(prop_value.referencevalue())
    if prop_value.has_uservalue():
        result = []
        uservalue = prop_value.uservalue()
        if uservalue.has_email():
            result.append((entity_pb.PropertyValue.kUserValueemail, uservalue.email()))
        if uservalue.has_auth_domain():
            result.append((entity_pb.PropertyValue.kUserValueauth_domain, uservalue.auth_domain()))
        if uservalue.has_nickname():
            result.append((entity_pb.PropertyValue.kUserValuenickname, uservalue.nickname()))
        if uservalue.has_gaiaid():
            result.append((entity_pb.PropertyValue.kUserValuegaiaid, uservalue.gaiaid()))
        if uservalue.has_obfuscated_gaiaid():
            result.append((entity_pb.PropertyValue.kUserValueobfuscated_gaiaid, uservalue.obfuscated_gaiaid()))
        if uservalue.has_federated_identity():
            result.append((entity_pb.PropertyValue.kUserValuefederated_identity, uservalue.federated_identity()))
        if uservalue.has_federated_provider():
            result.append((entity_pb.PropertyValue.kUserValuefederated_provider, uservalue.federated_provider()))
        result.sort()
        return (entity_pb.PropertyValue.kUserValueGroup, tuple(result))
    return ()