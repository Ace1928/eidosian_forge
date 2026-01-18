import sys
Indicates whether the authenticator implementation is available to sign.

    The caller should not call Authenticate() if IsAvailable() returns False

    Returns:
      True if the authenticator is available to sign and False otherwise.

    