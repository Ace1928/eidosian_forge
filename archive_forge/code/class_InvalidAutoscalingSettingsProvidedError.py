from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
class InvalidAutoscalingSettingsProvidedError(exceptions.Error):

    def __init__(self, details):
        super(InvalidAutoscalingSettingsProvidedError, self).__init__(f'INVALID_ARGUMENT: {details}')