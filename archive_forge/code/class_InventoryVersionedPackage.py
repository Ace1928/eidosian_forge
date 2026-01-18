from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InventoryVersionedPackage(_messages.Message):
    """Information related to the a standard versioned package. This includes
  package info for APT, Yum, Zypper, and Googet package managers.

  Fields:
    architecture: The system architecture this package is intended for.
    packageName: The name of the package.
    version: The version of the package.
  """
    architecture = _messages.StringField(1)
    packageName = _messages.StringField(2)
    version = _messages.StringField(3)