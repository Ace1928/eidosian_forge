from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
def RewrittenError(self, resource_type, method):
    """Returns a copy of the error with a new resource type."""
    body = self.details['response'] if self.details else None
    return type(self)(self.base_error, resource_type, self.resource_identifier, body=body, user_help=self.user_help)