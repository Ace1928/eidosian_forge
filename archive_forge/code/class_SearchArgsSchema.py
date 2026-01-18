import base64
import email
from typing import Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.gmail.base import GmailBaseTool
from langchain_community.tools.gmail.utils import clean_email_body
class SearchArgsSchema(BaseModel):
    """Input for GetMessageTool."""
    message_id: str = Field(..., description='The unique ID of the email message, retrieved from a search.')