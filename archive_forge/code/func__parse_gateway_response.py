import os
import re
from typing import List
import yaml
from mlflow.exceptions import MlflowException
from mlflow.version import VERSION as __version__
def _parse_gateway_response(self, response):
    from mlflow.gateway import get_route
    route_type = get_route(self.model_route).route_type
    if route_type == 'llm/v1/completions':
        return response['choices'][0]['text']
    elif route_type == 'llm/v1/chat':
        return response['choices'][0]['message']['content']
    else:
        raise MlflowException(f'Error when parsing gateway response: Unsupported route type for _PromptlabModel: {route_type}')