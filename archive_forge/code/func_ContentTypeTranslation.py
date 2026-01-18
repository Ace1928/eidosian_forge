from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def ContentTypeTranslation(content_type):
    """Translate content type from gcloud format to API format.

  Args:
    content_type: the gcloud format of content_type

  Returns:
    cloudasset API format of content_type.
  """
    if content_type == 'resource':
        return 'RESOURCE'
    if content_type == 'iam-policy':
        return 'IAM_POLICY'
    if content_type == 'org-policy':
        return 'ORG_POLICY'
    if content_type == 'access-policy':
        return 'ACCESS_POLICY'
    if content_type == 'os-inventory':
        return 'OS_INVENTORY'
    if content_type == 'relationship':
        return 'RELATIONSHIP'
    return 'CONTENT_TYPE_UNSPECIFIED'