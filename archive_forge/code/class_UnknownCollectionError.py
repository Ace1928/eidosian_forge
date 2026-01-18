from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
class UnknownCollectionError(Error):

    def __init__(self, api_name, api_version, collection):
        super(UnknownCollectionError, self).__init__('Collection [{collection}] does not exist for [{api}] [{version}].'.format(collection=collection, api=api_name, version=api_version))