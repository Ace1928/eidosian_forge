from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from six.moves import range  # pylint: disable=redefined-builtin
def LimitResourcesToDisplay(resources, resource_limit=MAX_RESOURCE_TO_DISPLAY):
    if len(resources) > resource_limit:
        log.status.Print('Note: maximum of %s resources are shown, please use describe command to show all of the resources.' % resource_limit)
        resources = resources[:resource_limit]
    return resources