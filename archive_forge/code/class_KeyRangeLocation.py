from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyRangeLocation(_messages.Message):
    """Location information for a specific key-range of a sharded computation.
  Currently we only support UTF-8 character splits to simplify encoding into
  JSON.

  Fields:
    dataDisk: The name of the data disk where data for this range is stored.
      This name is local to the Google Cloud Platform project and uniquely
      identifies the disk within that project, for example
      "myproject-1014-104817-4c2-harness-0-disk-1".
    deliveryEndpoint: The physical location of this range assignment to be
      used for streaming computation cross-worker message delivery.
    deprecatedPersistentDirectory: DEPRECATED. The location of the persistent
      state for this range, as a persistent directory in the worker local
      filesystem.
    end: The end (exclusive) of the key range.
    start: The start (inclusive) of the key range.
  """
    dataDisk = _messages.StringField(1)
    deliveryEndpoint = _messages.StringField(2)
    deprecatedPersistentDirectory = _messages.StringField(3)
    end = _messages.StringField(4)
    start = _messages.StringField(5)