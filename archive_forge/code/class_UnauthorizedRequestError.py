from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
class UnauthorizedRequestError(RequestError):
    """Raised when a request to the Apigee API had insufficient privileges."""

    def __init__(self, resource_type=None, resource_identifier=None, method=None, reason=None, body=None, message=None, user_help=None):
        resource_type = resource_type or 'resource'
        method = method or 'access'
        if not message:
            message = 'Insufficient privileges to %s the requested %s' % (method, resource_type)
            if reason:
                message += '; ' + reason
            if resource_identifier:
                message += '\nRequested: ' + _GetResourceIdentifierString(resource_type, resource_identifier)
            if user_help:
                message += '\n' + user_help
        super(UnauthorizedRequestError, self).__init__(resource_type, resource_identifier, method, reason, body, message, user_help)