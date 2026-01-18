from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def _GetParameterColumn(self, parameter_info, parameter_name):
    """Get this updater's column number for a certain parameter."""
    updater_parameters = self._GetRuntimeParameters(parameter_info)
    for parameter in updater_parameters:
        if parameter.name == parameter_name:
            return parameter.column
    return None