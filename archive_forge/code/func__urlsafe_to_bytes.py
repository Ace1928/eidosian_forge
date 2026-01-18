from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
@staticmethod
def _urlsafe_to_bytes(cursor):
    if not isinstance(cursor, six_subset.string_types + (six_subset.binary_type,)):
        raise datastore_errors.BadValueError('cursor argument should be str or unicode (%r)' % (cursor,))
    try:
        decoded_bytes = base64.urlsafe_b64decode(six_subset.ensure_binary(cursor, 'ascii'))
    except (ValueError, TypeError) as e:
        raise datastore_errors.BadValueError('Invalid cursor %s. Details: %s' % (cursor, e))
    return decoded_bytes