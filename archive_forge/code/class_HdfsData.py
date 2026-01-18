from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HdfsData(_messages.Message):
    """An HdfsData resource specifies a path within an HDFS entity (e.g. a
  cluster). All cluster-specific settings, such as namenodes and ports, are
  configured on the transfer agents servicing requests, so HdfsData only
  contains the root path to the data in our transfer.

  Fields:
    path: Root path to transfer files.
  """
    path = _messages.StringField(1)