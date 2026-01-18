from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _ConstructPrompt(apis_not_enabled: List[str]) -> str:
    """Returns a prompt to enable APIs with any custom text per-API.

  Args:
    apis_not_enabled: APIs that are to be enabled.
  Returns: prompt string to be displayed for confirmation.
  """
    if not apis_not_enabled:
        return ''
    base_prompt = 'Do you want to enable these APIs to continue (this will take a few minutes)?'
    prompt = ''
    for api in apis_not_enabled:
        if api in _API_ENABLEMENT_CONFIRMATION_TEXT:
            prompt += _API_ENABLEMENT_CONFIRMATION_TEXT[api] + '\n'
    prompt += base_prompt
    return prompt