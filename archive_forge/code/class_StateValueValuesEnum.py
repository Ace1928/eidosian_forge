from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateValueValuesEnum(_messages.Enum):
    """The current state of the CryptoKeyVersion.

    Values:
      CRYPTO_KEY_VERSION_STATE_UNSPECIFIED: Not specified.
      ENABLED: This version may be used in Encrypt and Decrypt requests.
      DISABLED: This version may not be used, but the key material is still
        available, and the version can be placed back into the ENABLED state.
      DESTROYED: This version is destroyed, and the key material is no longer
        stored. A version may not leave this state once entered.
      DESTROY_SCHEDULED: This version is scheduled for destruction, and will
        be destroyed soon. Call RestoreCryptoKeyVersion to put it back into
        the DISABLED state.
    """
    CRYPTO_KEY_VERSION_STATE_UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2
    DESTROYED = 3
    DESTROY_SCHEDULED = 4