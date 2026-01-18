from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.core import properties
def _MakeGetUriFunc(registry):
    """Returns a transformation function from a patch job resource to an URI."""

    def UriFunc(resource):
        ref = registry.Parse(resource.name, params={'projects': properties.VALUES.core.project.GetOrFail, 'patchJobs': resource.name}, collection='osconfig.projects.patchJobs')
        return ref.SelfLink()
    return UriFunc