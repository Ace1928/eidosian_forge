from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import urlencode
from .. import errors
from .base import BaseEndpoint
Validate an access token request.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :raises: OAuth1Error if the request is invalid.
        :returns: A tuple of 2 elements.
                  1. The validation result (True or False).
                  2. The request object.
        