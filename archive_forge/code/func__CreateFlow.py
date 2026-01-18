from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _CreateFlow(self):
    from googlecloudsdk.core.credentials import flow as c_flow
    try:
        return c_flow.FullWebFlow.from_client_config(self._client_config, self._scopes, autogenerate_code_verifier=not properties.VALUES.auth.disable_code_verifier.GetBool())
    except c_flow.LocalServerCreationError as e:
        log.warning(e)
        log.warning('Defaulting to --no-browser mode.')
        return c_flow.NoBrowserFlow.from_client_config(self._client_config, self._scopes, autogenerate_code_verifier=not properties.VALUES.auth.disable_code_verifier.GetBool())