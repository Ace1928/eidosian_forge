import hashlib
import hmac
import json
import os
import posixpath
import re
from six.moves import http_client
from six.moves import urllib
from six.moves.urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
def _get_signing_key(key, date_stamp, region_name, service_name):
    """Calculates the signing key used to calculate the signature for
    AWS Signature Version 4 based on:
    https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html

    Args:
        key (str): The AWS secret access key.
        date_stamp (str): The '%Y%m%d' date format.
        region_name (str): The AWS region.
        service_name (str): The AWS service name, eg. sts.

    Returns:
        str: The signing key bytes.
    """
    k_date = _sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = _sign(k_date, region_name)
    k_service = _sign(k_region, service_name)
    k_signing = _sign(k_service, 'aws4_request')
    return k_signing