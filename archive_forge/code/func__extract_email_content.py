import base64
import re
from typing import Any, Iterator
from langchain_core._api.deprecation import deprecated
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
def _extract_email_content(msg: Any) -> HumanMessage:
    from_email = None
    for values in msg['payload']['headers']:
        name = values['name']
        if name == 'From':
            from_email = values['value']
    if from_email is None:
        raise ValueError
    for part in msg['payload']['parts']:
        if part['mimeType'] == 'text/plain':
            data = part['body']['data']
            data = base64.urlsafe_b64decode(data).decode('utf-8')
            pattern = re.compile('\\r\\nOn .+(\\r\\n)*wrote:\\r\\n')
            newest_response = re.split(pattern, data)[0]
            message = HumanMessage(content=newest_response, additional_kwargs={'sender': from_email})
            return message
    raise ValueError