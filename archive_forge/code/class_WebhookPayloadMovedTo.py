from typing import List, Literal, Optional
from pydantic import BaseModel
class WebhookPayloadMovedTo(BaseModel):
    name: str
    owner: ObjectId