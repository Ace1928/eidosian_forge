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
def ResolveAppId(app):
    """Validate app id, providing a default.

  If the argument is None, $APPLICATION_ID is substituted.

  Args:
    app: The app id argument value to be validated.

  Returns:
    The value of app, or the substituted default.  Always a non-empty string.

  Raises:
    BadArgumentError if the value is empty or not a string.
  """
    if app is None:
        app = os.environ.get('APPLICATION_ID', '')
    ValidateString(app, 'app', datastore_errors.BadArgumentError)
    return app