from typing import Any, Dict, List, Optional
import requests
from langchain_core.messages import get_buffer_string
from langchain.memory.chat_memory import BaseChatMemory
def __get_headers(self) -> Dict[str, str]:
    is_managed = self.url == MANAGED_URL
    headers = {'Content-Type': 'application/json'}
    if is_managed and (not (self.api_key and self.client_id)):
        raise ValueError('\n                You must provide an API key or a client ID to use the managed\n                version of Motorhead. Visit https://getmetal.io for more information.\n                ')
    if is_managed and self.api_key and self.client_id:
        headers['x-metal-api-key'] = self.api_key
        headers['x-metal-client-id'] = self.client_id
    return headers