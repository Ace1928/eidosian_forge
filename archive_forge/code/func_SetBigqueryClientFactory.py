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
def SetBigqueryClientFactory(cls, factory):
    if not issubclass(factory, bigquery_client.BigqueryClient):
        raise TypeError('Factory must be subclass of BigqueryClient.')
    cls._BIGQUERY_CLIENT_FACTORY = factory