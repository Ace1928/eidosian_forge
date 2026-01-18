from typing import Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
def _send_sms(self, sender: str, destination_phone_numbers: List[str], text: str) -> str:
    """Send an SMS message."""
    json: Dict = {'messages': [{'destinations': [{'to': destination} for destination in destination_phone_numbers], 'from': sender, 'text': text}]}
    session: requests.Session = self._get_requests_session()
    session.headers.update({'Content-Type': 'application/json'})
    response: requests.Response = session.post(f'{self.infobip_base_url}/sms/2/text/advanced', json=json)
    response_json: Dict = response.json()
    try:
        if response.status_code != 200:
            return response_json['requestError']['serviceException']['text']
    except KeyError:
        return 'Failed to send message'
    try:
        return response_json['messages'][0]['messageId']
    except KeyError:
        return 'Could not get message ID from response, message was sent successfully'