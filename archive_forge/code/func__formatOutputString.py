from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from absl import app
from absl import flags
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def _formatOutputString(self, table_reference: bq_id_utils.ApiClientHelper.TableReference, status: str) -> str:
    return '%s %200s' % (table_reference, status)