from __future__ import absolute_import, unicode_literals
import json
import logging
from .. import errors
from ..request_validator import RequestValidator
from .base import GrantTypeBase

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        