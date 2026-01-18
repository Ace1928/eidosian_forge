from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from clients import bigquery_client_extended
from frontend import utils as frontend_utils
def CheckValidCreds(reference, data_source, transfer_client):
    """Checks valid credentials.

  Checks if Data Transfer Service valid credentials exist for the given data
  source and requesting user. Some data sources don't support service account,
  so we need to talk to them on behalf of the end user. This method just checks
  whether we have OAuth token for the particular user, which is a pre-requisite
  before a user can create a transfer config.

  Args:
    reference: The project reference.
    data_source: The data source of the transfer config.
    transfer_client: The transfer api client.

  Returns:
    credentials: It contains an instance of CheckValidCredsResponse if valid
    credentials exist.
  """
    credentials = None
    if FLAGS.location:
        data_source_reference = reference + '/locations/' + FLAGS.location + '/dataSources/' + data_source
        credentials = transfer_client.projects().locations().dataSources().checkValidCreds(name=data_source_reference, body={}).execute()
    else:
        data_source_reference = reference + '/dataSources/' + data_source
        credentials = transfer_client.projects().dataSources().checkValidCreds(name=data_source_reference, body={}).execute()
    return credentials