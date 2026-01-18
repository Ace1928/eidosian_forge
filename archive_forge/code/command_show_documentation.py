from typing import Optional
from absl import app
from absl import flags
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as bq_frontend_utils
from utils import bq_id_utils
Show all information about an object.

    Examples:
      bq show -j <job_id>
      bq show dataset
      bq show [--schema] dataset.table
      bq show [--view] dataset.view
      bq show [--materialized_view] dataset.materialized_view
      bq show -m ds.model
      bq show --routine ds.routine
      bq show --transfer_config projects/p/locations/l/transferConfigs/c
      bq show --transfer_run projects/p/locations/l/transferConfigs/c/runs/r
      bq show --encryption_service_account
      bq show --connection --project_id=project --location=us connection
      bq show --capacity_commitment project:US.capacity_commitment_id
      bq show --reservation --location=US --project_id=project reservation_name
      bq show --reservation_assignment --project_id=project --location=US
          --assignee_type=PROJECT --assignee_id=myproject --job_type=QUERY
      bq show --reservation_assignment --project_id=project --location=US
          --assignee_type=FOLDER --assignee_id=123 --job_type=QUERY
      bq show --reservation_assignment --project_id=project --location=US
          --assignee_type=ORGANIZATION --assignee_id=456 --job_type=QUERY

    Arguments:
      identifier: the identifier of the resource to show.
    