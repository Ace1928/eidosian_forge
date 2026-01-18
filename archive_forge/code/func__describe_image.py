from typing import Dict
import requests
from langchain_core.pydantic_v1 import BaseModel, BaseSettings, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _describe_image(self, image: str) -> str:
    headers = {'x-api-key': f'token {self.scenex_api_key}', 'content-type': 'application/json'}
    payload = {'data': [{'image': image, 'algorithm': 'Ember', 'languages': ['en']}]}
    response = requests.post(self.scenex_api_url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json().get('result', [])
    img = result[0] if result else {}
    return img.get('text', '')