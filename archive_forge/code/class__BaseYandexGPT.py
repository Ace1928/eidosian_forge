from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
from langchain_community.llms.utils import enforce_stop_tokens
class _BaseYandexGPT(Serializable):
    iam_token: SecretStr = ''
    'Yandex Cloud IAM token for service or user account\n    with the `ai.languageModels.user` role'
    api_key: SecretStr = ''
    'Yandex Cloud Api Key for service account\n    with the `ai.languageModels.user` role'
    folder_id: str = ''
    'Yandex Cloud folder ID'
    model_uri: str = ''
    'Model uri to use.'
    model_name: str = 'yandexgpt-lite'
    'Model name to use.'
    model_version: str = 'latest'
    'Model version to use.'
    temperature: float = 0.6
    'What sampling temperature to use.\n    Should be a double number between 0 (inclusive) and 1 (inclusive).'
    max_tokens: int = 7400
    'Sets the maximum limit on the total number of tokens\n    used for both the input prompt and the generated response.\n    Must be greater than zero and not exceed 7400 tokens.'
    stop: Optional[List[str]] = None
    'Sequences when completion generation will stop.'
    url: str = 'llm.api.cloud.yandex.net:443'
    'The url of the API.'
    max_retries: int = 6
    'Maximum number of retries to make when generating.'
    sleep_interval: float = 1.0
    'Delay between API requests'
    _grpc_metadata: Sequence

    @property
    def _llm_type(self) -> str:
        return 'yandex_gpt'

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {'model_uri': self.model_uri, 'temperature': self.temperature, 'max_tokens': self.max_tokens, 'stop': self.stop, 'max_retries': self.max_retries}

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that iam token exists in environment."""
        iam_token = convert_to_secret_str(get_from_dict_or_env(values, 'iam_token', 'YC_IAM_TOKEN', ''))
        values['iam_token'] = iam_token
        api_key = convert_to_secret_str(get_from_dict_or_env(values, 'api_key', 'YC_API_KEY', ''))
        values['api_key'] = api_key
        folder_id = get_from_dict_or_env(values, 'folder_id', 'YC_FOLDER_ID', '')
        values['folder_id'] = folder_id
        if api_key.get_secret_value() == '' and iam_token.get_secret_value() == '':
            raise ValueError("Either 'YC_API_KEY' or 'YC_IAM_TOKEN' must be provided.")
        if values['iam_token']:
            values['_grpc_metadata'] = [('authorization', f'Bearer {values['iam_token'].get_secret_value()}')]
            if values['folder_id']:
                values['_grpc_metadata'].append(('x-folder-id', values['folder_id']))
        else:
            values['_grpc_metadata'] = (('authorization', f'Api-Key {values['api_key'].get_secret_value()}'),)
        if values['model_uri'] == '' and values['folder_id'] == '':
            raise ValueError("Either 'model_uri' or 'folder_id' must be provided.")
        if not values['model_uri']:
            values['model_uri'] = f'gpt://{values['folder_id']}/{values['model_name']}/{values['model_version']}'
        return values