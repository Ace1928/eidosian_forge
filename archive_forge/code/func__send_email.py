from typing import Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
def _send_email(self, from_email: str, to_email: str, subject: str, body: str) -> str:
    """Send an email message."""
    try:
        from requests_toolbelt import MultipartEncoder
    except ImportError as e:
        raise ImportError('Unable to import requests_toolbelt, please install it with `pip install -U requests-toolbelt`.') from e
    form_data: Dict = {'from': from_email, 'to': to_email, 'subject': subject, 'text': body}
    data = MultipartEncoder(fields=form_data)
    session: requests.Session = self._get_requests_session()
    session.headers.update({'Content-Type': data.content_type})
    response: requests.Response = session.post(f'{self.infobip_base_url}/email/3/send', data=data)
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