import logging
import os
import urllib.parse
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import REQUEST_URL_CHAT
def _call_gateway_api(gateway_uri, payload, eval_parameters):
    from mlflow.gateway import get_route, query
    route_info = get_route(gateway_uri).dict()
    if route_info['endpoint_type'] == 'llm/v1/completions':
        completions_payload = {'prompt': payload, **eval_parameters}
        response = query(gateway_uri, completions_payload)
        return _parse_completions_response_format(response)
    elif route_info['endpoint_type'] == 'llm/v1/chat':
        chat_payload = {'messages': [{'role': 'user', 'content': payload}], **eval_parameters}
        response = query(gateway_uri, chat_payload)
        return _parse_chat_response_format(response)
    else:
        raise MlflowException(f"Unsupported gateway route type: {route_info['endpoint_type']}. Use a route of type 'llm/v1/completions' or 'llm/v1/chat' instead.", error_code=INVALID_PARAMETER_VALUE)