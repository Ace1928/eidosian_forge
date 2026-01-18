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
def _verify_iat_and_exp(payload, clock_skew_in_seconds=0):
    """Verifies the ``iat`` (Issued At) and ``exp`` (Expires) claims in a token
    payload.

    Args:
        payload (Mapping[str, str]): The JWT payload.
        clock_skew_in_seconds (int): The clock skew used for `iat` and `exp`
            validation.

    Raises:
        google.auth.exceptions.InvalidValue: if value validation failed.
        google.auth.exceptions.MalformedError: if schema validation failed.
    """
    now = _helpers.datetime_to_secs(_helpers.utcnow())
    for key in ('iat', 'exp'):
        if key not in payload:
            raise exceptions.MalformedError('Token does not contain required claim {}'.format(key))
    iat = payload['iat']
    earliest = iat - clock_skew_in_seconds
    if now < earliest:
        raise exceptions.InvalidValue("Token used too early, {} < {}. Check that your computer's clock is set correctly.".format(now, iat))
    exp = payload['exp']
    latest = exp + clock_skew_in_seconds
    if latest < now:
        raise exceptions.InvalidValue('Token expired, {} < {}'.format(latest, now))