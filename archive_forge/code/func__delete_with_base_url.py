import abc
import contextlib
import hashlib
import os
from cinderclient.apiclient import base as common_base
from cinderclient import exceptions
from cinderclient import utils
def _delete_with_base_url(self, url, response_key=None):
    self.api.client.delete_with_base_url(url)