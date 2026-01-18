import logging
import os
import urllib.parse
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import REQUEST_URL_CHAT
def _call_deployments_api(deployment_uri, payload, eval_parameters, wrap_payload=True):
    """Call the deployment endpoint with the given payload and parameters.

    Args:
        deployment_uri: The URI of the deployment endpoint.
        payload: The input payload to send to the endpoint.
        eval_parameters: The evaluation parameters to send to the endpoint.
        wrap_payload: Whether to wrap the payload in a expected key by the endpoint,
            e.g. "prompt" for completions or "messages" for chat. If False, the specified
            payload is directly sent to the endpoint combined with the eval_parameters.

    Returns:
        The unpacked response from the endpoint.
    """
    from pydantic import BaseModel
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client()
    endpoint = client.get_endpoint(deployment_uri)
    endpoint = endpoint.dict() if isinstance(endpoint, BaseModel) else endpoint
    endpoint_type = endpoint.get('task', endpoint.get('endpoint_type'))
    if endpoint_type == 'llm/v1/completions':
        if wrap_payload:
            payload = {'prompt': payload}
        chat_inputs = {**payload, **eval_parameters}
        response = client.predict(endpoint=deployment_uri, inputs=chat_inputs)
        return _parse_completions_response_format(response)
    elif endpoint_type == 'llm/v1/chat':
        if wrap_payload:
            payload = {'messages': [{'role': 'user', 'content': payload}]}
        completion_inputs = {**payload, **eval_parameters}
        response = client.predict(endpoint=deployment_uri, inputs=completion_inputs)
        return _parse_chat_response_format(response)
    else:
        raise MlflowException(f"Unsupported endpoint type: {endpoint_type}. Use an endpoint of type 'llm/v1/completions' or 'llm/v1/chat' instead.", error_code=INVALID_PARAMETER_VALUE)