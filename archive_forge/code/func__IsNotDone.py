from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.features import info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
def _IsNotDone(feature, unused_state):
    if feature.state is None or feature.state.state is None:
        return True
    return feature.state.state.code != ok_code