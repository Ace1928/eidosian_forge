from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _GetCatalog(client, messages, environment_type):
    """Gets a test environment catalog from the TestEnvironmentDiscoveryService.

  Args:
    client: The Testing API client object.
    messages: The Testing API messages object.
    environment_type: {enum} which EnvironmentType catalog to get.

  Returns:
    The test environment catalog.

  Raises:
    calliope_exceptions.HttpException: If it could not connect to the service.
  """
    project_id = properties.VALUES.core.project.Get()
    request = messages.TestingTestEnvironmentCatalogGetRequest(environmentType=environment_type, projectId=project_id)
    try:
        return client.testEnvironmentCatalog.Get(request)
    except apitools_exceptions.HttpError as error:
        raise calliope_exceptions.HttpException('Unable to access the test environment catalog: ' + GetError(error))
    except:
        log.error('Unable to access the Firebase Test Lab environment catalog.')
        raise