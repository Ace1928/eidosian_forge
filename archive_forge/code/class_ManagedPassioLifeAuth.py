from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, final
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
class ManagedPassioLifeAuth(NoDiskStorage):
    """Manages the token for the NutritionAI API."""
    _access_token_expiry: Optional[datetime]

    def __init__(self, subscription_key: str):
        self.subscription_key = subscription_key
        self._last_token = None
        self._access_token_expiry = None
        self._access_token = None
        self._customer_id = None

    @property
    def headers(self) -> dict:
        if not self.is_valid_now():
            self.refresh_access_token()
        return {'Authorization': f'Bearer {self._access_token}', 'Passio-ID': self._customer_id}

    def is_valid_now(self) -> bool:
        return self._access_token is not None and self._customer_id is not None and (self._access_token_expiry is not None) and (self._access_token_expiry > datetime.now())

    @retry(retry=retry_if_result(is_http_retryable), stop=stop_after_attempt(4), wait=wait_random(0, 0.3) + wait_exponential(multiplier=1, min=0.1, max=2))
    def _http_get(self, subscription_key: str) -> requests.Response:
        return requests.get(f'https://api.passiolife.com/v2/token-cache/napi/oauth/token/{subscription_key}')

    def refresh_access_token(self) -> None:
        """Refresh the access token for the NutritionAI API."""
        rsp = self._http_get(self.subscription_key)
        if not rsp:
            raise ValueError('Could not get access token')
        self._last_token = token = rsp.json()
        self._customer_id = token['customer_id']
        self._access_token = token['access_token']
        self._access_token_expiry = datetime.now() + timedelta(seconds=token['expires_in']) - timedelta(seconds=5)