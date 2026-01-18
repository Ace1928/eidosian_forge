import time
from typing import Any, Dict
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import MistralConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import completions, embeddings
class MistralProvider(BaseProvider):
    NAME = 'Mistral'

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MistralConfig):
            raise TypeError(f'Unexpected config type {config.model.config}')
        self.mistral_config: MistralConfig = config.model.config

    @property
    def auth_headers(self) -> Dict[str, str]:
        return {'Authorization': f'Bearer {self.mistral_config.mistral_api_key}'}

    @property
    def base_url(self) -> str:
        return 'https://api.mistral.ai/v1/'

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await send_request(headers=self.auth_headers, base_url=self.base_url, path=path, payload=payload)

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request('chat/completions', {'model': self.config.model.name, **MistralAdapter.completions_to_model(payload, self.config)})
        return MistralAdapter.model_to_completions(resp, self.config)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request('embeddings', {'model': self.config.model.name, **MistralAdapter.embeddings_to_model(payload, self.config)})
        return MistralAdapter.model_to_embeddings(resp, self.config)