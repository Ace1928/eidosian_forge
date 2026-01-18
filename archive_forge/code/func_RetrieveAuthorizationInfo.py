from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from clients import bigquery_client_extended
from frontend import utils as frontend_utils
def RetrieveAuthorizationInfo(reference, data_source, transfer_client):
    """Retrieves the authorization code.

  An authorization code is needed if the Data Transfer Service does not
  have credentials for the requesting user and data source. The Data
  Transfer Service will convert this authorization code into a refresh
  token to perform transfer runs on the user's behalf.

  Args:
    reference: The project reference.
    data_source: The data source of the transfer config.
    transfer_client: The transfer api client.

  Returns:
    auth_info: A dict which contains authorization info from user. It is either
    an authorization_code or a version_info.
  """
    data_source_retrieval = reference + '/dataSources/' + data_source
    data_source_info = transfer_client.projects().dataSources().get(name=data_source_retrieval).execute()
    first_party_oauth = False
    if data_source_info['authorizationType'] == 'FIRST_PARTY_OAUTH':
        first_party_oauth = True
    auth_uri = 'https://www.gstatic.com/bigquerydatatransfer/oauthz/auth?client_id=' + data_source_info['clientId'] + '&scope=' + '%20'.join(data_source_info['scopes']) + '&redirect_uri=urn:ietf:wg:oauth:2.0:oob&response_type=' + ('version_info' if first_party_oauth else 'authorization_code')
    print('\n' + auth_uri)
    auth_info = {}
    if first_party_oauth:
        print('Please copy and paste the above URL into your web browser and follow the instructions to retrieve a version_info.')
        auth_info[bigquery_client_extended.VERSION_INFO] = frontend_utils.RawInput('Enter your version_info here: ')
    else:
        print('Please copy and paste the above URL into your web browser and follow the instructions to retrieve an authorization code.')
        auth_info[bigquery_client_extended.AUTHORIZATION_CODE] = frontend_utils.RawInput('Enter your authorization code here: ')
    return auth_info