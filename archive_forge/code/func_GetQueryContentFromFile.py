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
def GetQueryContentFromFile(self, file_path):
    """Returns a message populated from the JSON or YAML file on the specified filepath."""
    file_content = yaml.load_path(file_path)
    try:
        query_type_str = next(iter(file_content.keys()))
    except:
        raise gcloud_exceptions.BadFileException('Query file [{0}] is not a properly formatted YAML or JSON query file. Supported query type: {1}.'.format(file_path, self.DictKeysToString(self.supported_query_types.keys())))
    if query_type_str not in self.supported_query_types.keys():
        raise Exception('query type {0} not supported. supported query type: {1}'.format(query_type_str, self.DictKeysToString(self.supported_query_types.keys())))
    query_content = file_content[query_type_str]
    try:
        query_obj = encoding.PyValueToMessage(self.supported_query_types[query_type_str], query_content)
    except:
        raise gcloud_exceptions.BadFileException('Query file [{0}] is not a properly formatted YAML or JSON query file.'.format(file_path))
    return query_obj