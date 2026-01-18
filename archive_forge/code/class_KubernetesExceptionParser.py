from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import exceptions
import six
class KubernetesExceptionParser(object):
    """Converts a kubernetes exception to an object."""

    def __init__(self, http_error):
        """Wraps a generic http error returned by kubernetes.

    Args:
      http_error: apitools.base.py.exceptions.HttpError, The error to wrap.
    """
        self._wrapped_error = http_error
        self._content = json.loads(http_error.content)

    @property
    def status_code(self):
        try:
            return self._wrapped_error.status_code
        except KeyError:
            return None

    @property
    def url(self):
        return self._wrapped_error.url

    @property
    def api_version(self):
        try:
            return self._content['apiVersion']
        except KeyError:
            return None

    @property
    def api_name(self):
        try:
            return self._content['details']['group']
        except KeyError:
            return None

    @property
    def resource_name(self):
        try:
            return self._content['details']['name']
        except KeyError:
            return None

    @property
    def resource_kind(self):
        try:
            return self._content['details']['kind']
        except KeyError:
            return None

    @property
    def default_message(self):
        try:
            return self._content['message']
        except KeyError:
            return None

    @property
    def error(self):
        return self._wrapped_error

    @property
    def causes(self):
        """Returns list of causes uniqued by the message."""
        try:
            messages = {c['message']: c for c in self._content['details']['causes']}
            return [messages[k] for k in sorted(messages)]
        except KeyError:
            return []