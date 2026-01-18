import base64
import re
from typing import Any, Iterator
from langchain_core._api.deprecation import deprecated
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
def _get_message_data(service: Any, message: Any) -> ChatSession:
    msg = service.users().messages().get(userId='me', id=message['id']).execute()
    message_content = _extract_email_content(msg)
    in_reply_to = None
    email_data = msg['payload']['headers']
    for values in email_data:
        name = values['name']
        if name == 'In-Reply-To':
            in_reply_to = values['value']
    if in_reply_to is None:
        raise ValueError
    thread_id = msg['threadId']
    thread = service.users().threads().get(userId='me', id=thread_id).execute()
    messages = thread['messages']
    response_email = None
    for message in messages:
        email_data = message['payload']['headers']
        for values in email_data:
            if values['name'] == 'Message-ID':
                message_id = values['value']
                if message_id == in_reply_to:
                    response_email = message
    if response_email is None:
        raise ValueError
    starter_content = _extract_email_content(response_email)
    return ChatSession(messages=[starter_content, message_content])