from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.ml_engine import region_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def ParseModel(model):
    """Parses a model ID into a model resource object."""
    return resources.REGISTRY.Parse(model, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection=MODELS_COLLECTION)