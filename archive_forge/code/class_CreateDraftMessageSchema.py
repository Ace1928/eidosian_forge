from typing import List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.office365.base import O365BaseTool
class CreateDraftMessageSchema(BaseModel):
    """Input for SendMessageTool."""
    body: str = Field(..., description='The message body to include in the draft.')
    to: List[str] = Field(..., description='The list of recipients.')
    subject: str = Field(..., description='The subject of the message.')
    cc: Optional[List[str]] = Field(None, description='The list of CC recipients.')
    bcc: Optional[List[str]] = Field(None, description='The list of BCC recipients.')