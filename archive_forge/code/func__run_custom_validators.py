from __future__ import absolute_import, unicode_literals
import logging
from oauthlib import common
from .. import errors
from .base import GrantTypeBase
def _run_custom_validators(self, request, validations, request_info=None):
    request_info = {} if request_info is None else request_info.copy()
    for validator in validations:
        result = validator(request)
        if result is not None:
            request_info.update(result)
    return request_info