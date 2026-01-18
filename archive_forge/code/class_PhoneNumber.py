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
class PhoneNumber(six_subset.text_type):
    """A human-readable phone number or address.

  No validation is performed. Phone numbers have many different formats -
  local, long distance, domestic, international, internal extension, TTY,
  VOIP, SMS, and alternative networks like Skype, XFire and Roger Wilco. They
  all have their own numbering and addressing formats.

  This is the gd:phoneNumber element. In XML output, the phone number is
  provided as the text of the element. See:
  https://developers.google.com/gdata/docs/1.0/elements#gdPhoneNumber

  Raises BadValueError if phone is not a string or subtype.
  """

    def __init__(self, phone):
        super(PhoneNumber, self).__init__()
        ValidateString(phone, 'phone')

    def ToXml(self):
        return u'<gd:phoneNumber>%s</gd:phoneNumber>' % saxutils.escape(self)