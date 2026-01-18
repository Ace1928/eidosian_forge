import base64
import email
from enum import Enum
from typing import Any, Dict, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.gmail.base import GmailBaseTool
from langchain_community.tools.gmail.utils import clean_email_body
def _parse_threads(self, threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for thread in threads:
        thread_id = thread['id']
        thread_data = self.api_resource.users().threads().get(userId='me', id=thread_id).execute()
        messages = thread_data['messages']
        thread['messages'] = []
        for message in messages:
            snippet = message['snippet']
            thread['messages'].append({'snippet': snippet, 'id': message['id']})
        results.append(thread)
    return results