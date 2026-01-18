import base64
import email
from enum import Enum
from typing import Any, Dict, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.gmail.base import GmailBaseTool
from langchain_community.tools.gmail.utils import clean_email_body
def _parse_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for message in messages:
        message_id = message['id']
        message_data = self.api_resource.users().messages().get(userId='me', format='raw', id=message_id).execute()
        raw_message = base64.urlsafe_b64decode(message_data['raw'])
        email_msg = email.message_from_bytes(raw_message)
        subject = email_msg['Subject']
        sender = email_msg['From']
        message_body = ''
        if email_msg.is_multipart():
            for part in email_msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    try:
                        message_body = part.get_payload(decode=True).decode('utf-8')
                    except UnicodeDecodeError:
                        message_body = part.get_payload(decode=True).decode('latin-1')
                    break
        else:
            message_body = email_msg.get_payload(decode=True).decode('utf-8')
        body = clean_email_body(message_body)
        results.append({'id': message['id'], 'threadId': message_data['threadId'], 'snippet': message_data['snippet'], 'body': body, 'subject': subject, 'sender': sender})
    return results