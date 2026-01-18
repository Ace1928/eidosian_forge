from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def CreateConnection(self, project_id, location, connection_type, properties, connection_credential=None, display_name=None, description=None, connection_id=None, kms_key_name=None, connector_configuration=None):
    """Create a connection with the given connection reference.

    Arguments:
      project_id: Project ID.
      location: Location of connection.
      connection_type: Type of connection, allowed values: ['CLOUD_SQL']
      properties: Connection properties in JSON format.
      connection_credential: Connection credentials in JSON format.
      display_name: Friendly name for the connection.
      description: Description of the connection.
      connection_id: Optional connection ID.
      kms_key_name: Optional KMS key name.
      connector_configuration: Optional configuration for connector.

    Returns:
      Connection object that was created.
    """
    connection = {}
    if display_name:
        connection['friendlyName'] = display_name
    if description:
        connection['description'] = description
    if kms_key_name:
        connection['kmsKeyName'] = kms_key_name
    property_name = bq_client_utils.CONNECTION_TYPE_TO_PROPERTY_MAP.get(connection_type)
    if property_name:
        connection[property_name] = bq_processor_utils.ParseJson(properties)
        if connection_credential:
            connection[property_name]['credential'] = bq_processor_utils.ParseJson(connection_credential)
    elif connector_configuration:
        connection['configuration'] = bq_processor_utils.ParseJson(connector_configuration)
    else:
        error = 'connection_type %s is unsupported or connector_configuration is not specified' % connection_type
        raise ValueError(error)
    client = self.GetConnectionV1ApiClient()
    parent = 'projects/%s/locations/%s' % (project_id, location)
    return client.projects().locations().connections().create(parent=parent, connectionId=connection_id, body=connection).execute()