from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegesRequiredValueValuesEnum(_messages.Enum):
    """This metric describes the level of privileges an attacker must possess
    before successfully exploiting the vulnerability.

    Values:
      PRIVILEGES_REQUIRED_UNSPECIFIED: Invalid value.
      PRIVILEGES_REQUIRED_NONE: The attacker is unauthorized prior to attack,
        and therefore does not require any access to settings or files of the
        vulnerable system to carry out an attack.
      PRIVILEGES_REQUIRED_LOW: The attacker requires privileges that provide
        basic user capabilities that could normally affect only settings and
        files owned by a user. Alternatively, an attacker with Low privileges
        has the ability to access only non-sensitive resources.
      PRIVILEGES_REQUIRED_HIGH: The attacker requires privileges that provide
        significant (e.g., administrative) control over the vulnerable
        component allowing access to component-wide settings and files.
    """
    PRIVILEGES_REQUIRED_UNSPECIFIED = 0
    PRIVILEGES_REQUIRED_NONE = 1
    PRIVILEGES_REQUIRED_LOW = 2
    PRIVILEGES_REQUIRED_HIGH = 3