from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
from typing import Optional
from absl import app
from absl import flags
from pyglib import appcommands
import bq_utils
from clients import bigquery_client_extended
from clients import client_dataset
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import flags as frontend_flags
from frontend import utils as frontend_utils
from frontend import utils_data_transfer
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
Create a dataset, table, view, or transfer configuration with this name.

    See 'bq help mk' for more information.

    Examples:
      bq mk new_dataset
      bq mk new_dataset.new_table
      bq --dataset_id=new_dataset mk table
      bq mk -t new_dataset.newtable name:integer,value:string
      bq mk --view='select 1 as num' new_dataset.newview
         (--view_udf_resource=path/to/file.js)
      bq mk --materialized_view='select sum(x) as sum_x from dataset.table'
          new_dataset.newview
      bq mk -d --data_location=EU new_dataset
      bq mk -d --source_dataset=src_dataset new_dataset (requires allowlisting)
      bq mk -d
        --external_source=aws-glue://<aws_arn_of_glue_database>
        --connection_id=<connection>
        new_dataset
      bq mk --transfer_config --target_dataset=dataset --display_name=name
          -p='{"param":"value"}' --data_source=source
          --schedule_start_time={schedule_start_time}
          --schedule_end_time={schedule_end_time}
      bq mk --transfer_run --start_time={start_time} --end_time={end_time}
          projects/p/locations/l/transferConfigs/c
      bq mk --transfer_run --run_time={run_time}
          projects/p/locations/l/transferConfigs/c
      bq mk --reservation --project_id=project --location=us reservation_name
      bq mk --reservation_assignment --reservation_id=project:us.dev
          --job_type=QUERY --assignee_type=PROJECT --assignee_id=myproject
      bq mk --reservation_assignment --reservation_id=project:us.dev
          --job_type=QUERY --assignee_type=FOLDER --assignee_id=123
      bq mk --reservation_assignment --reservation_id=project:us.dev
          --job_type=QUERY --assignee_type=ORGANIZATION --assignee_id=456
      bq mk --connection --connection_type='CLOUD_SQL'
        --properties='{"instanceId" : "instance",
        "database" : "db", "type" : "MYSQL" }'
        --connection_credential='{"username":"u", "password":"p"}'
        --project_id=proj --location=us --display_name=name new_connection
    