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
class Email(six_subset.text_type):
    """An RFC2822 email address. Makes no attempt at validation; apart from
  checking MX records, email address validation is a rathole.

  This is the gd:email element. In XML output, the email address is provided as
  the address attribute. See:
  https://developers.google.com/gdata/docs/1.0/elements#gdEmail

  Raises BadValueError if email is not a valid email address.
  """

    def __init__(self, email):
        super(Email, self).__init__()
        ValidateString(email, 'email')

    def ToXml(self):
        return u'<gd:email address=%s />' % saxutils.quoteattr(self)