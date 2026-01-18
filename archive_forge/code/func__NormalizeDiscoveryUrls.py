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
def _NormalizeDiscoveryUrls(discovery_url):
    """Expands a few abbreviations into full discovery urls."""
    if discovery_url.startswith('http'):
        return [discovery_url]
    elif '.' not in discovery_url:
        raise ValueError('Unrecognized value "%s" for discovery url')
    api_name, _, api_version = discovery_url.partition('.')
    return ['https://www.googleapis.com/discovery/v1/apis/%s/%s/rest' % (api_name, api_version), 'https://%s.googleapis.com/$discovery/rest?version=%s' % (api_name, api_version)]