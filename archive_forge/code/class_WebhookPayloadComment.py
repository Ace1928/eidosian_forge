from typing import List, Literal, Optional
from pydantic import BaseModel
class WebhookPayloadComment(ObjectId):
    author: ObjectId
    hidden: bool
    content: Optional[str] = None
    url: WebhookPayloadUrl