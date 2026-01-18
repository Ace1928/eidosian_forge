from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, final
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
@retry(retry=retry_if_result(is_http_retryable), stop=stop_after_attempt(4), wait=wait_random(0, 0.3) + wait_exponential(multiplier=1, min=0.1, max=2))
def _http_get(self, params: dict) -> requests.Response:
    return requests.get(self.nutritionai_api_url, headers=self.auth_.headers, params=params)