from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import time
from typing import Dict, List, Optional
from absl import app
from absl import flags
from pyglib import appcommands
import bq_utils
from clients import bigquery_client_extended
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from frontend import utils_data_transfer
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
Updates a dataset, table, view or transfer configuration with this name.

    See 'bq help update' for more information.

    Examples:
      bq update --description "Dataset description" existing_dataset
      bq update --description "My table" existing_dataset.existing_table
      bq update --description "My model" -m existing_dataset.existing_model
      bq update -t existing_dataset.existing_table name:integer,value:string
      bq update --destination_kms_key
          projects/p/locations/l/keyRings/r/cryptoKeys/k
          existing_dataset.existing_table
      bq update --view='select 1 as num' existing_dataset.existing_view
         (--view_udf_resource=path/to/file.js)
      bq update --transfer_config --display_name=name -p='{"param":"value"}'
          projects/p/locations/l/transferConfigs/c
      bq update --transfer_config --target_dataset=dataset
          --refresh_window_days=5 --update_credentials
          projects/p/locations/l/transferConfigs/c
      bq update --reservation --location=US --project_id=my-project
          --bi_reservation_size=2G
      bq update --capacity_commitment --location=US --project_id=my-project
          --plan=MONTHLY --renewal_plan=FLEX commitment_id
      bq update --capacity_commitment --location=US --project_id=my-project
        --split --slots=500 commitment_id
      bq update --capacity_commitment --location=US --project_id=my-project
        --merge commitment_id1,commitment_id2
      bq update --reservation_assignment
          --destination_reservation_id=proj:US.new_reservation
          proj:US.old_reservation.assignment_id
      bq update --connection_credential='{"username":"u", "password":"p"}'
        --location=US --project_id=my-project existing_connection
    