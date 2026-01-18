from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepositoryBaseValueValuesEnum(_messages.Enum):
    """A common public repository base for Yum.

    Values:
      REPOSITORY_BASE_UNSPECIFIED: Unspecified repository base.
      CENTOS: CentOS.
      CENTOS_DEBUG: CentOS Debug.
      CENTOS_VAULT: CentOS Vault.
      CENTOS_STREAM: CentOS Stream.
      ROCKY: Rocky.
      EPEL: Fedora Extra Packages for Enterprise Linux (EPEL).
    """
    REPOSITORY_BASE_UNSPECIFIED = 0
    CENTOS = 1
    CENTOS_DEBUG = 2
    CENTOS_VAULT = 3
    CENTOS_STREAM = 4
    ROCKY = 5
    EPEL = 6