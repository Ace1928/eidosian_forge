import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict
import aiohttp
import requests
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
def _kwargs_post_fine_tune_request(self, inputs: Sequence[str], kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build the kwargs for the Post request, used by sync

        Args:
            prompt (str): prompt used in query
            kwargs (dict): model kwargs in payload

        Returns:
            Dict[str, Union[str,dict]]: _description_
        """
    _model_kwargs = self.model_kwargs or {}
    _params = {**_model_kwargs, **kwargs}
    multipliers = _params.get('multipliers', None)
    return dict(url=f'{self.gradient_api_url}/models/{self.model_id}/fine-tune', headers={'authorization': f'Bearer {self.gradient_access_token}', 'x-gradient-workspace-id': f'{self.gradient_workspace_id}', 'accept': 'application/json', 'content-type': 'application/json'}, json=dict(samples=tuple(({'inputs': input} for input in inputs)) if multipliers is None else tuple(({'inputs': input, 'fineTuningParameters': {'multiplier': multiplier}} for input, multiplier in zip(inputs, multipliers)))))