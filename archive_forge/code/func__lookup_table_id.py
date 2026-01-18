import logging
from typing import Any, Dict, Optional
from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_managed_catalog_messages_pb2 import (
from mlflow.protos.databricks_managed_catalog_service_pb2 import UnityCatalogService
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import get_full_name_from_sc
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
def _lookup_table_id(self, table_name):
    try:
        req_body = message_to_json(GetTable(full_name_arg=table_name))
        _METHOD_TO_INFO = extract_api_info_for_service(UnityCatalogService, _REST_API_PATH_PREFIX)
        db_creds = get_databricks_host_creds()
        endpoint, method = _METHOD_TO_INFO[GetTable]
        final_endpoint = endpoint.replace('{full_name_arg}', table_name)
        resp = call_endpoint(host_creds=db_creds, endpoint=final_endpoint, method=method, json_body=req_body, response_proto=GetTableResponse)
        return resp.table_id
    except Exception:
        return None