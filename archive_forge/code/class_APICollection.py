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
class APICollection(object):
    """A data holder for collection information for an API."""

    def __init__(self, collection_info):
        self.api_name = collection_info.api_name
        self.api_version = collection_info.api_version
        self.base_url = collection_info.base_url
        self.docs_url = collection_info.docs_url
        self.name = collection_info.name
        self.full_name = collection_info.full_name
        self.detailed_path = collection_info.GetPath('')
        self.detailed_params = collection_info.GetParams('')
        self.path = collection_info.path
        self.params = collection_info.params
        self.enable_uri_parsing = collection_info.enable_uri_parsing