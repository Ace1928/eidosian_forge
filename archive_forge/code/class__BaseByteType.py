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
class _BaseByteType(str):
    """A base class for datastore types that are encoded as bytes.

  This behaves identically to the Python str type, except for the
  constructor, which only accepts str arguments.
  """

    def __new__(cls, arg=None):
        """Constructor.

    We only accept str instances.

    Args:
      arg: optional str instance (default '')
    """
        if arg is None:
            arg = ''
        if isinstance(arg, str):
            return super(_BaseByteType, cls).__new__(cls, arg)
        raise TypeError('%s() argument should be str instance, not %s' % (cls.__name__, type(arg).__name__))

    def ToXml(self):
        """Output bytes as XML.

    Returns:
      Base64 encoded version of itself for safe insertion in to an XML document.
    """
        encoded = base64.urlsafe_b64encode(self)
        return saxutils.escape(encoded)