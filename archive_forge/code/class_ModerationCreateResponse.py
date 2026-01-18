from typing import List
from .._models import BaseModel
from .moderation import Moderation
class ModerationCreateResponse(BaseModel):
    id: str
    'The unique identifier for the moderation request.'
    model: str
    'The model used to generate the moderation results.'
    results: List[Moderation]
    'A list of moderation objects.'