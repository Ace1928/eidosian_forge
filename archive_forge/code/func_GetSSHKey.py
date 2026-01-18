from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_printer
import six
def GetSSHKey(self, product, resource):
    """Make a Get SSH Key request. Return details of specified SSH key."""
    _ValidateProduct(product)
    resource = resource.RelativeName()
    if product == _PFORG:
        power_request = self.messages.MarketplacesolutionsProjectsLocationsPowerSshKeysGetRequest(name=resource)
        return self.power_sshkeys_service.Get(power_request)