import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Type, Union
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.gmail.base import GmailBaseTool
def _prepare_message(self, message: str, to: Union[str, List[str]], subject: str, cc: Optional[Union[str, List[str]]]=None, bcc: Optional[Union[str, List[str]]]=None) -> Dict[str, Any]:
    """Create a message for an email."""
    mime_message = MIMEMultipart()
    mime_message.attach(MIMEText(message, 'html'))
    mime_message['To'] = ', '.join(to if isinstance(to, list) else [to])
    mime_message['Subject'] = subject
    if cc is not None:
        mime_message['Cc'] = ', '.join(cc if isinstance(cc, list) else [cc])
    if bcc is not None:
        mime_message['Bcc'] = ', '.join(bcc if isinstance(bcc, list) else [bcc])
    encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
    return {'raw': encoded_message}