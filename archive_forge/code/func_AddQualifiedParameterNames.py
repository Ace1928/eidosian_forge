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
def AddQualifiedParameterNames(self, qualified_parameter_names):
    """Adds qualified_parameter_names to the set of qualified parameters."""
    self.qualified_parameter_names |= qualified_parameter_names