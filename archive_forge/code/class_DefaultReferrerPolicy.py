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
class DefaultReferrerPolicy(NoReferrerWhenDowngradePolicy):
    """
    A variant of "no-referrer-when-downgrade",
    with the addition that "Referer" is not sent if the parent request was
    using ``file://`` or ``s3://`` scheme.
    """
    NOREFERRER_SCHEMES: Tuple[str, ...] = LOCAL_SCHEMES + ('file', 's3')
    name: str = POLICY_SCRAPY_DEFAULT