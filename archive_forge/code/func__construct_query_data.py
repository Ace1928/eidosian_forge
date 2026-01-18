import os
import re
from typing import List
import yaml
from mlflow.exceptions import MlflowException
from mlflow.version import VERSION as __version__
def _construct_query_data(self, prompt):
    from mlflow.gateway import get_route
    route_type = get_route(self.model_route).route_type
    if route_type == 'llm/v1/completions':
        return {'prompt': prompt}
    elif route_type == 'llm/v1/chat':
        return {'messages': [{'content': prompt, 'role': 'user'}]}
    else:
        raise MlflowException(f'Error when constructing gateway query: Unsupported route type for _PromptlabModel: {route_type}')