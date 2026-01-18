import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
class _DatabricksServingEndpointClient(_DatabricksClientBase):
    """An API client that talks to a Databricks serving endpoint."""
    host: str
    endpoint_name: str
    databricks_uri: str
    client: Any = None
    external_or_foundation: bool = False
    task: Optional[str] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        try:
            from mlflow.deployments import get_deploy_client
            self.client = get_deploy_client(self.databricks_uri)
        except ImportError as e:
            raise ImportError('Failed to create the client. Please install mlflow with `pip install mlflow`.') from e
        endpoint = self.client.get_endpoint(self.endpoint_name)
        self.external_or_foundation = endpoint.get('endpoint_type', '').lower() in ('external_model', 'foundation_model_api')
        if self.task is None:
            self.task = endpoint.get('task')

    @property
    def llm(self) -> bool:
        return self.task in ('llm/v1/chat', 'llm/v1/completions', 'llama2/chat')

    @root_validator(pre=True)
    def set_api_url(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if 'api_url' not in values:
            host = values['host']
            endpoint_name = values['endpoint_name']
            api_url = f'https://{host}/serving-endpoints/{endpoint_name}/invocations'
            values['api_url'] = api_url
        return values

    def post(self, request: Any, transform_output_fn: Optional[Callable[..., str]]=None) -> Any:
        if self.external_or_foundation:
            resp = self.client.predict(endpoint=self.endpoint_name, inputs=request)
            if transform_output_fn:
                return transform_output_fn(resp)
            if self.task == 'llm/v1/chat':
                return _transform_chat(resp)
            elif self.task == 'llm/v1/completions':
                return _transform_completions(resp)
            return resp
        else:
            wrapped_request = {'dataframe_records': [request]}
            response = self.client.predict(endpoint=self.endpoint_name, inputs=wrapped_request)
            preds = response['predictions']
            pred = preds[0] if isinstance(preds, list) else preds
            if self.task == 'llama2/chat':
                return _transform_llama2_chat(pred)
            return transform_output_fn(pred) if transform_output_fn else pred