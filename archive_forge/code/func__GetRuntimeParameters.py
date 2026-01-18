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
def _GetRuntimeParameters(self, parameter_info):
    """Constructs and returns the _RuntimeParameter list.

    This method constructs a muable shadow of self.parameters with updater_class
    and table instantiations. Each runtime parameter can be:

    (1) A static value derived from parameter_info.
    (2) A parameter with it's own updater_class.  The updater is used to list
        all of the possible values for the parameter.
    (3) An unknown value (None).  The possible values are contained in the
        resource cache for self.

    The Select method combines the caller supplied row template and the runtime
    parameters to filter the list of parsed resources in the resource cache.

    Args:
      parameter_info: A ParamaterInfo object for accessing parameter values in
        the program state.

    Returns:
      The runtime parameters shadow of the immutable self.parameters.
    """
    runtime_parameters = []
    for parameter in self.parameters:
        updater_class, aggregator = parameter_info.GetUpdater(parameter.name)
        value = parameter_info.GetValue(parameter.name, check_properties=aggregator)
        runtime_parameter = _RuntimeParameter(parameter, updater_class, value, aggregator)
        runtime_parameters.append(runtime_parameter)
    return runtime_parameters