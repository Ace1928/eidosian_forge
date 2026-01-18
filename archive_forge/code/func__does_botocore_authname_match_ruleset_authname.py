import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
def _does_botocore_authname_match_ruleset_authname(self, botoname, rsname):
    """
        Whether a valid string provided as signature_version parameter for
        client construction refers to the same auth methods as a string
        returned by the endpoint ruleset provider. This accounts for:

        * The ruleset prefixes auth names with "sig"
        * The s3 and s3control rulesets don't distinguish between v4[a] and
          s3v4[a] signers
        * The v2, v3, and HMAC v1 based signers (s3, s3-*) are botocore legacy
          features and do not exist in the rulesets
        * Only characters up to the first dash are considered

        Example matches:
        * v4, sigv4
        * v4, v4
        * s3v4, sigv4
        * s3v7, sigv7 (hypothetical example)
        * s3v4a, sigv4a
        * s3v4-query, sigv4

        Example mismatches:
        * v4a, sigv4
        * s3, sigv4
        * s3-presign-post, sigv4
        """
    rsname = self._strip_sig_prefix(rsname)
    botoname = botoname.split('-')[0]
    if botoname != 's3' and botoname.startswith('s3'):
        botoname = botoname[2:]
    return rsname == botoname