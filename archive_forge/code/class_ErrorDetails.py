from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
import ruamel.yaml as ryaml
class ErrorDetails(object):
    """Data class for ErrorDetail sub-messages."""
    _DEFAULT_ERROR_FORMAT = '[{error}].'
    _DEFAULT_CONTEXT_FORMAT = ' Additional details: [{context}]'

    def __init__(self, error_msg, context=None, as_json=False, level='error'):
        self.error = error_msg
        self.context = context
        self.as_json = as_json
        self.level = level
        if not self.level:
            self.level = 'error'

    def AsDict(self):
        out = collections.OrderedDict(error=self.error)
        if self.context:
            out['context'] = self.context
        return out

    def __str__(self):
        if self.as_json:
            return json.dumps(self.AsDict())
        return yaml.dump(self.AsDict(), round_trip=True)

    def __eq__(self, other):
        if not isinstance(other, OutputMessage.ErrorDetails):
            return False
        return self.error == other.error and self.context == other.context

    def Format(self, error_format=None, context_format=None):
        """Render formatted ErrorDetails string."""
        output_string = error_format or self._DEFAULT_ERROR_FORMAT
        output_string = output_string.format(error=self.error, level=self.level.capitalize())
        if self.context:
            context_string = context_format or self._DEFAULT_CONTEXT_FORMAT
            context_string = context_string.format(context=self.context)
            output_string += context_string
        return output_string