from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import model_monitoring_jobs_util
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _ParseDataset(dataset_id, region_ref):
    """Parses a dataset ID into a dataset resource object."""
    region = region_ref.AsDict()['locationsId']
    return resources.REGISTRY.Parse(dataset_id, params={'locationsId': region, 'projectsId': properties.VALUES.core.project.GetOrFail}, collection='aiplatform.projects.locations.datasets')