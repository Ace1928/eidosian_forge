import os
import re
from typing import List
import yaml
from mlflow.exceptions import MlflowException
from mlflow.version import VERSION as __version__
class _PromptlabModel:
    import pandas as pd

    def __init__(self, prompt_template, prompt_parameters, model_parameters, model_route):
        self.prompt_parameters = prompt_parameters
        self.model_parameters = model_parameters
        self.model_route = model_route
        self.prompt_template = prompt_template

    def predict(self, inputs: pd.DataFrame) -> List[str]:
        from mlflow.gateway import query
        results = []
        for idx in inputs.index:
            prompt_parameters_as_dict = {param.key: inputs[param.key][idx] for param in self.prompt_parameters}
            prompt = self.prompt_template
            for key, value in prompt_parameters_as_dict.items():
                prompt = re.sub('\\{\\{\\s*' + key + '\\s*\\}\\}', value, prompt)
            model_parameters_as_dict = {param.key: param.value for param in self.model_parameters}
            query_data = self._construct_query_data(prompt)
            response = query(route=self.model_route, data={**query_data, **model_parameters_as_dict})
            results.append(self._parse_gateway_response(response))
        return results

    def _construct_query_data(self, prompt):
        from mlflow.gateway import get_route
        route_type = get_route(self.model_route).route_type
        if route_type == 'llm/v1/completions':
            return {'prompt': prompt}
        elif route_type == 'llm/v1/chat':
            return {'messages': [{'content': prompt, 'role': 'user'}]}
        else:
            raise MlflowException(f'Error when constructing gateway query: Unsupported route type for _PromptlabModel: {route_type}')

    def _parse_gateway_response(self, response):
        from mlflow.gateway import get_route
        route_type = get_route(self.model_route).route_type
        if route_type == 'llm/v1/completions':
            return response['choices'][0]['text']
        elif route_type == 'llm/v1/chat':
            return response['choices'][0]['message']['content']
        else:
            raise MlflowException(f'Error when parsing gateway response: Unsupported route type for _PromptlabModel: {route_type}')