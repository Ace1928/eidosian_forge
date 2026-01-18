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
class ParameterInfo(object):
    """An object for accessing parameter values in the program state.

  "program state" is defined by this class.  It could include parsed command
  line arguments and properties.  The class also can also map between resource
  and program parameter names.

  Attributes:
    _additional_params: The list of parameter names not in the parsed resource.
    _updaters: A parameter_name => (Updater, aggregator) dict.
  """

    def __init__(self, additional_params=None, updaters=None):
        self._additional_params = additional_params or []
        self._updaters = updaters or {}

    def GetValue(self, parameter_name, check_properties=True):
        """Returns the program state string value for parameter_name.

    Args:
      parameter_name: The Parameter name.
      check_properties: Check the property value if True.

    Returns:
      The parameter value from the program state.
    """
        del parameter_name, check_properties
        return None

    def GetAdditionalParams(self):
        """Return the list of parameter names not in the parsed resource.

    These names are associated with the resource but not a specific parameter
    in the resource.  For example a global resource might not have a global
    Boolean parameter in the parsed resource, but its command line specification
    might require a --global flag to completly qualify the resource.

    Returns:
      The list of parameter names not in the parsed resource.
    """
        return self._additional_params

    def GetUpdater(self, parameter_name):
        """Returns the updater and aggregator property for parameter_name.

    Args:
      parameter_name: The Parameter name.

    Returns:
      An (updater, aggregator) tuple where updater is the Updater class and
      aggregator is True if this updater must be used to aggregate all resource
      values.
    """
        return self._updaters.get(parameter_name, (None, None))