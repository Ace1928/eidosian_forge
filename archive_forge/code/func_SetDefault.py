from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def SetDefault(versions_client, version, model=None):
    version_ref = ParseVersion(model, version)
    return versions_client.SetDefault(version_ref)