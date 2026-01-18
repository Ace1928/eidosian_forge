from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import gce as c_gce
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def CreateRegionalReference(self, resource_name, region_arg, flag_names=None, resource_type=None):
    return self.CreateRegionalReferences([resource_name], region_arg, flag_names, resource_type)[0]