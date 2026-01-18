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
class NoBrowserHelperRunner(FlowRunner):
    """A flow runner to run NoBrowserHelperFlow."""

    def _CreateFlow(self):
        from googlecloudsdk.core.credentials import flow as c_flow
        try:
            return c_flow.NoBrowserHelperFlow.from_client_config(self._client_config, self._scopes, autogenerate_code_verifier=not properties.VALUES.auth.disable_code_verifier.GetBool())
        except c_flow.LocalServerCreationError:
            log.error('Cannot start a local server to handle authorization redirection. Please run this command on a machine where gcloud can start a local server.')
            raise