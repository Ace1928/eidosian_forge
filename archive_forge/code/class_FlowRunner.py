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
class FlowRunner(six.with_metaclass(abc.ABCMeta, object)):
    """Base auth flow runner class.

  Attributes:
     _scopes: [str], The list of scopes to authorize.
     _client_config: The client configuration in the Google client secrets
       format.
  """
    _FLOW_ERROR_HELP_MSG = 'There was a problem with web authentication.'

    def __init__(self, scopes, client_config, redirect_uri=None):
        self._scopes = scopes
        self._client_config = client_config
        self._redirect_uri = redirect_uri
        self._flow = self._CreateFlow()

    @abc.abstractmethod
    def _CreateFlow(self):
        pass

    def Run(self, **kwargs):
        from googlecloudsdk.core.credentials import flow as c_flow
        try:
            return self._flow.Run(**kwargs)
        except c_flow.Error as e:
            _HandleFlowError(e, self._FLOW_ERROR_HELP_MSG)
            raise