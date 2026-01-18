from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
class GcsHmacKeyResource:
    """Holds HMAC key metadata."""

    def __init__(self, metadata):
        self.metadata = metadata

    @property
    def access_id(self):
        key_metadata = getattr(self.metadata, 'metadata', None)
        return getattr(key_metadata, 'accessId', None)

    @property
    def secret(self):
        return getattr(self.metadata, 'secret', None)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.metadata == other.metadata