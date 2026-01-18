from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
def _AssertValidResource(arg, resource_name):
    if not any([resource_name.startswith(t) for t in ('projects/', 'organizations/', 'folders/', 'billingAccounts/')]):
        raise exceptions.InvalidArgumentException(arg, 'Invalid resource %s. Resource must be in the form [projects|folders|organizations|billingAccounts]/{{resource_id}}' % resource_name)