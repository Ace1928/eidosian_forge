from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.core.console import console_io
def ShouldAttemptProjectEnable(project):
    return project not in _PROJECTS_NOT_TO_ENABLE