import warnings
from typing import Tuple
from urllib.parse import urlparse
from w3lib.url import safe_url_string
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.http import Request, Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_unicode
from scrapy.utils.url import strip_url
def _load_policy_class(policy, warning_only=False):
    """
    Expect a string for the path to the policy class,
    otherwise try to interpret the string as a standard value
    from https://www.w3.org/TR/referrer-policy/#referrer-policies
    """
    try:
        return load_object(policy)
    except ValueError:
        try:
            return _policy_classes[policy.lower()]
        except KeyError:
            msg = f'Could not load referrer policy {policy!r}'
            if not warning_only:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg, RuntimeWarning)
                return None