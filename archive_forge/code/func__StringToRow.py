from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.api_lib.util import resource_search
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import completion_cache
from googlecloudsdk.core.cache import resource_cache
import six
def _StringToRow(self, string, parameter_info=None):
    if string and (string.startswith('https://') or string.startswith('http://') or self._parse_all):
        try:
            row = self.parse(string or None)
            if parameter_info:
                self._ConvertProjectNumberToID(row, parameter_info)
            row = list(row.values())
            return row
        except resources.RequiredFieldOmittedException:
            pass
    return [''] * (self.columns - 1) + [string]