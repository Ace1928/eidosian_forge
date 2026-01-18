import abc
import contextlib
import hashlib
import os
from cinderclient.apiclient import base as common_base
from cinderclient import exceptions
from cinderclient import utils
def _get_all_with_base_url(self, url, response_key=None):
    resp, body = self.api.client.get_with_base_url(url)
    if response_key:
        if isinstance(body[response_key], list):
            return [self.resource_class(self, res, loaded=True) for res in body[response_key] if res]
        return self.resource_class(self, body[response_key], loaded=True)
    return self.resource_class(self, body, loaded=True)