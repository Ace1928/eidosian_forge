from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import time
from typing import Optional
from absl import flags
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import flags as frontend_flags
from frontend import utils as frontend_utils
Perform a load operation of source into destination_table.

    Usage:
      load <destination_table> <source> [<schema>] [--session_id=[session]]

    The <destination_table> is the fully-qualified table name of table to
    create, or append to if the table already exists.

    To load to a temporary table, specify the table name in <destination_table>
    without a dataset and specify the session id with --session_id.

    The <source> argument can be a path to a single local file, or a
    comma-separated list of URIs.

    The <schema> argument should be either the name of a JSON file or a text
    schema. This schema should be omitted if the table already has one.

    In the case that the schema is provided in text form, it should be a
    comma-separated list of entries of the form name[:type], where type will
    default to string if not specified.

    In the case that <schema> is a filename, it should be a JSON file
    containing a single array, each entry of which should be an object with
    properties 'name', 'type', and (optionally) 'mode'. For more detail:
    https://cloud.google.com/bigquery/docs/schemas#specifying_a_json_schema_file

    Note: the case of a single-entry schema with no type specified is
    ambiguous; one can use name:string to force interpretation as a
    text schema.

    Examples:
      bq load ds.new_tbl ./info.csv ./info_schema.json
      bq load ds.new_tbl gs://mybucket/info.csv ./info_schema.json
      bq load ds.small gs://mybucket/small.csv name:integer,value:string
      bq load ds.small gs://mybucket/small.csv field1,field2,field3
      bq load temp_tbl --session_id=my_session ./info.csv ./info_schema.json

    Arguments:
      destination_table: Destination table name.
      source: Name of local file to import, or a comma-separated list of URI
        paths to data to import.
      schema: Either a text schema or JSON file, as above.
    