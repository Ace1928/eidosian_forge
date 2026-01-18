from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RootsyncCrdValueValuesEnum(_messages.Enum):
    """The state of the RootSync CRD

    Values:
      CRD_STATE_UNSPECIFIED: CRD's state cannot be determined
      NOT_INSTALLED: CRD is not installed
      INSTALLED: CRD is installed
      TERMINATING: CRD is terminating (i.e., it has been deleted and is
        cleaning up)
      INSTALLING: CRD is installing
    """
    CRD_STATE_UNSPECIFIED = 0
    NOT_INSTALLED = 1
    INSTALLED = 2
    TERMINATING = 3
    INSTALLING = 4