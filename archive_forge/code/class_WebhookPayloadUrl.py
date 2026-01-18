from typing import List, Literal, Optional
from pydantic import BaseModel
class WebhookPayloadUrl(BaseModel):
    web: str
    api: Optional[str] = None