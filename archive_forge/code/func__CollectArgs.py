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
@staticmethod
def _CollectArgs(config_logging=True, **kwds):
    """Collect and combine FLAGS and kwds to create BQ Client.

    Args:
      config_logging: if True, set python logging according to --apilog.
      **kwds: keyword arguments for creating BigqueryClient.
    """

    def KwdsOrFlags(name):
        return kwds[name] if name in kwds else getattr(FLAGS, name)
    bq_utils.ProcessBigqueryrc()
    if config_logging:
        bq_logging.ConfigureLogging(bq_flags.APILOG.value)
    bq_utils.ProcessGcloudConfig(flag_values=FLAGS)
    if bq_flags.UNIVERSE_DOMAIN.present and (not bq_auth_flags.USE_GOOGLE_AUTH.value):
        raise app.UsageError('Attempting to use TPC without setting `use_google_auth`.')
    if FLAGS.httplib2_debuglevel:
        httplib2.debuglevel = FLAGS.httplib2_debuglevel
    client_args = {}
    global_args = ('credential_file', 'job_property', 'project_id', 'dataset_id', 'trace', 'sync', 'use_google_auth', 'api', 'api_version', 'quota_project_id')
    for name in global_args:
        client_args[name] = KwdsOrFlags(name)
    logging.debug('Global args collected: %s', client_args)
    client_args['wait_printer_factory'] = _GetWaitPrinterFactoryFromFlags()
    if FLAGS.discovery_file:
        with open(FLAGS.discovery_file) as f:
            client_args['discovery_document'] = f.read()
    client_args['enable_resumable_uploads'] = True if FLAGS.enable_resumable_uploads is None else FLAGS.enable_resumable_uploads
    if FLAGS.max_rows_per_request:
        client_args['max_rows_per_request'] = FLAGS.max_rows_per_request
    logging.info('Client args collected: %s', client_args)
    return client_args