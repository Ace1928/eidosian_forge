from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class StsEndpoints(ByoidEndpoints):
    """Simple class to build STS endpoints."""

    def __init__(self, **kwargs):
        super(StsEndpoints, self).__init__('sts', **kwargs)

    @property
    def token_url(self):
        api = 'v1/token'
        return '{}/{}'.format(self._base_url, api)

    @property
    def oauth_token_url(self):
        api = 'v1/oauthtoken'
        return '{}/{}'.format(self._base_url, api)

    @property
    def token_info_url(self):
        api = 'v1/introspect'
        return '{}/{}'.format(self._base_url, api)