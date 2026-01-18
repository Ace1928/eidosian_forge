from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def RetrieveMutatingWebhookURL(self, revision):
    """Retrieves the Mutating Webhook URL used for a revision."""
    if revision == 'default':
        incluster_webhook = _INCLUSTER_WEBHOOK_PREFIX
    else:
        incluster_webhook = '{}-{}'.format(_INCLUSTER_WEBHOOK_PREFIX, revision)
    mcp_webhook = '{}-{}'.format(_MCP_WEBHOOK_PREFIX, revision)
    incluster_out, incluster_err = self._RunKubectl(['get', 'mutatingwebhookconfiguration', incluster_webhook, '-o', 'jsonpath={.webhooks[0].clientConfig.url}'], None)
    mcp_out, mcp_err = self._RunKubectl(['get', 'mutatingwebhookconfiguration', mcp_webhook, '-o', 'jsonpath={.webhooks[0].clientConfig.url}'], None)
    if incluster_err and mcp_err:
        if 'NotFound' in incluster_err and 'NotFound' in mcp_err:
            raise ClusterError('Anthos Service Mesh revision {} is not found in the cluster. Please install Anthos Service Mesh and try again.'.format(revision))
        raise exceptions.Error('Error retrieving the mutating webhook configuration from the cluster: {}'.format(incluster_err))
    if incluster_out:
        return incluster_out
    return mcp_out