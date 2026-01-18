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
class AssetSavedQueriesClient(object):
    """Client for asset saved queries."""

    def DictKeysToString(self, keys):
        return ', '.join(list(keys))

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

    def __init__(self, parent, api_version=DEFAULT_API_VERSION):
        self.parent = parent
        self.message_module = GetMessages(api_version)
        self.service = GetClient(api_version).savedQueries
        self.supported_query_types = {'IamPolicyAnalysisQuery': self.message_module.IamPolicyAnalysisQuery}

    def Create(self, args):
        """Create a SavedQuery."""
        query_obj = self.GetQueryContentFromFile(args.query_file_path)
        saved_query_content = self.message_module.QueryContent(iamPolicyAnalysisQuery=query_obj)
        arg_labels = labels_util.ParseCreateArgs(args, self.message_module.SavedQuery.LabelsValue)
        saved_query = self.message_module.SavedQuery(content=saved_query_content, description=args.description, labels=arg_labels)
        request_message = self.message_module.CloudassetSavedQueriesCreateRequest(parent=self.parent, savedQuery=saved_query, savedQueryId=args.query_id)
        return self.service.Create(request_message)

    def Describe(self, args):
        """Describe a saved query."""
        request_message = self.message_module.CloudassetSavedQueriesGetRequest(name='{}/savedQueries/{}'.format(self.parent, args.query_id))
        return self.service.Get(request_message)

    def Delete(self, args):
        """Delete a saved query."""
        request_message = self.message_module.CloudassetSavedQueriesDeleteRequest(name='{}/savedQueries/{}'.format(self.parent, args.query_id))
        self.service.Delete(request_message)

    def List(self):
        """List saved queries under a parent."""
        request_message = self.message_module.CloudassetSavedQueriesListRequest(parent=self.parent)
        return self.service.List(request_message)

    def GetUpdatedLabels(self, args):
        """Get the updated labels from args."""
        labels_diff = labels_util.Diff.FromUpdateArgs(args)
        labels = self.message_module.SavedQuery.LabelsValue()
        if labels_diff.MayHaveUpdates():
            orig_resource = self.Describe(args)
            labels_update = labels_diff.Apply(self.message_module.SavedQuery.LabelsValue, orig_resource.labels)
            if labels_update.needs_update:
                labels = labels_update.labels
                return (labels, True)
        return (labels, False)

    def Update(self, args):
        """Update a saved query."""
        update_mask = ''
        saved_query_content = None
        if args.query_file_path:
            query_obj = self.GetQueryContentFromFile(args.query_file_path)
            update_mask += 'content'
            saved_query_content = self.message_module.QueryContent(iamPolicyAnalysisQuery=query_obj)
        updated_description = None
        if args.description:
            updated_description = args.description
            update_mask += ',description'
        updated_labels, has_update = self.GetUpdatedLabels(args)
        if has_update:
            update_mask += ',labels'
        saved_query = self.message_module.SavedQuery(content=saved_query_content, description=updated_description, labels=updated_labels)
        request_message = self.message_module.CloudassetSavedQueriesPatchRequest(name='{}/savedQueries/{}'.format(self.parent, args.query_id), savedQuery=saved_query, updateMask=update_mask)
        return self.service.Patch(request_message)