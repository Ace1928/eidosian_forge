import copy
import datetime
import json
import cachetools
import six
from six.moves import urllib
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials
@classmethod
def _from_signer_and_info(cls, signer, info, **kwargs):
    """Creates an OnDemandCredentials instance from a signer and service
        account info.

        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            info (Mapping[str, str]): The service account info.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.OnDemandCredentials: The constructed credentials.

        Raises:
            google.auth.exceptions.MalformedError: If the info is not in the expected format.
        """
    kwargs.setdefault('subject', info['client_email'])
    kwargs.setdefault('issuer', info['client_email'])
    return cls(signer, **kwargs)