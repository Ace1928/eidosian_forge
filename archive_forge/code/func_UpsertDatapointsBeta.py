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
def UpsertDatapointsBeta(self, index_ref, args):
    """Upsert data points from a v1beta1 index."""
    datapoints = []
    if args.datapoints_from_file:
        data = yaml.load_path(args.datapoints_from_file)
        for datapoint_json in data:
            datapoint = messages_util.DictToMessageWithErrorCheck(datapoint_json, self.messages.GoogleCloudAiplatformV1beta1IndexDatapoint)
            datapoints.append(datapoint)
    update_mask = None
    if args.update_mask:
        update_mask = ','.join(args.update_mask)
    req = self.messages.AiplatformProjectsLocationsIndexesUpsertDatapointsRequest(index=index_ref.RelativeName(), googleCloudAiplatformV1beta1UpsertDatapointsRequest=self.messages.GoogleCloudAiplatformV1beta1UpsertDatapointsRequest(datapoints=datapoints, updateMask=update_mask))
    return self._service.UpsertDatapoints(req)