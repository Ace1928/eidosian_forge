from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
def FetchDiscoveryDoc(discovery_url, retries=5):
    """Fetch the discovery document at the given url."""
    discovery_urls = _NormalizeDiscoveryUrls(discovery_url)
    discovery_doc = None
    last_exception = None
    for url in discovery_urls:
        for _ in range(retries):
            try:
                content = _GetURLContent(url)
                if isinstance(content, bytes):
                    content = content.decode('utf8')
                discovery_doc = json.loads(content)
                if discovery_doc:
                    return discovery_doc
            except (urllib_error.HTTPError, urllib_error.URLError) as e:
                logging.info('Attempting to fetch discovery doc again after "%s"', e)
                last_exception = e
    if discovery_doc is None:
        raise CommunicationError('Could not find discovery doc at any of %s: %s' % (discovery_urls, last_exception))