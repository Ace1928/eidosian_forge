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
def _GetTableInfo(self, table_reference: bq_id_utils.ApiClientHelper.TableReference):
    recovery_timestamp_for_table_query = "SELECT\n  TABLE_NAME,\n  UNIX_MILLIS(replicated_time_at_remote_site),\n  CASE\n    WHEN last_update_time <= min_latest_replicated_time THEN TRUE\n  ELSE\n  FALSE\nEND\n  AS fully_replicated\nFROM (\n  SELECT\n    TABLE_NAME,\n    multi_site_info.last_update_time,\n    ARRAY_AGG(site_info.latest_replicated_time\n    ORDER BY\n      latest_replicated_time DESC)[safe_OFFSET(1)] AS replicated_time_at_remote_site,\n    ARRAY_AGG(site_info.latest_replicated_time\n    ORDER BY\n      latest_replicated_time ASC)[safe_OFFSET(0)] AS min_latest_replicated_time\n  FROM\n    %s.INFORMATION_SCHEMA.TABLES t,\n    t.multi_site_info.site_info\n  WHERE\n    TABLE_NAME = '%s'\n  GROUP BY\n    1,\n    2 )" % (table_reference.datasetId, table_reference.tableId)
    return self._ReadTableInfo(recovery_timestamp_for_table_query, row_count=1)