import logging
import sys
import textwrap
from absl import app
from absl import flags
import httplib2
import termcolor
import bigquery_client
import bq_auth_flags
import bq_flags
import bq_utils
import credential_loader
from auth import main_credential_loader
from frontend import utils as bq_frontend_utils
from utils import bq_logging
@classmethod
def _GetClientCacheKey(cls, **kwds):
    logging.debug('In Client._GetClientCacheKey: %s', kwds)
    client_args = Client._CollectArgs(**kwds)
    return 'client_args={client_args},service_account_credential_file={service_account_credential_file},apilog={apilog},'.format(client_args=client_args, service_account_credential_file=FLAGS.service_account_credential_file, apilog=FLAGS.apilog)