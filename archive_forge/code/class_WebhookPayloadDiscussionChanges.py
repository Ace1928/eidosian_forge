from typing import List, Literal, Optional
from pydantic import BaseModel
class WebhookPayloadDiscussionChanges(BaseModel):
    base: str
    mergeCommitId: Optional[str] = None