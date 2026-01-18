from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def FallBackToDeployedProxyRevision(args):
    """If `args` provides no revision, adds the deployed revision, if unambiguous.

  Args:
    args: a dictionary of resource identifiers which identifies an API proxy and
      an environment, to which the deployed revision should be added.

  Raises:
    EntityNotFoundError: no deployment that matches `args` exists.
    AmbiguousRequestError: more than one deployment matches `args`.
  """
    deployments = apigee.DeploymentsClient.List(args)
    if not deployments:
        error_identifier = collections.OrderedDict([('organization', args['organizationsId']), ('environment', args['environmentsId']), ('api', args['apisId'])])
        raise errors.EntityNotFoundError('deployment', error_identifier, 'undeploy')
    if len(deployments) > 1:
        message = 'Found more than one deployment that matches this request.\n'
        raise errors.AmbiguousRequestError(message + yaml.dump(deployments))
    deployed_revision = deployments[0]['revision']
    log.status.Print('Using deployed revision `%s`' % deployed_revision)
    args['revisionsId'] = deployed_revision