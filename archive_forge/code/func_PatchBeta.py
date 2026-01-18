from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import yaml
def PatchBeta(self, index_ref, args):
    """Update an index."""
    index = self.messages.GoogleCloudAiplatformV1beta1Index()
    update_mask = []
    if args.metadata_file is not None:
        index.metadata = self._ReadIndexMetadata(args.metadata_file)
        update_mask.append('metadata')
    else:
        if args.display_name is not None:
            index.displayName = args.display_name
            update_mask.append('display_name')
        if args.description is not None:
            index.description = args.description
            update_mask.append('description')

        def GetLabels():
            return self.Get(index_ref).labels
        labels_update = labels_util.ProcessUpdateArgsLazy(args, self.messages.GoogleCloudAiplatformV1beta1Index.LabelsValue, GetLabels)
        if labels_update.needs_update:
            index.labels = labels_update.labels
            update_mask.append('labels')
    if not update_mask:
        raise errors.NoFieldsSpecifiedError('No updates requested.')
    request = self.messages.AiplatformProjectsLocationsIndexesPatchRequest(name=index_ref.RelativeName(), googleCloudAiplatformV1beta1Index=index, updateMask=','.join(update_mask))
    return self._service.Patch(request)