from typing_extensions import Literal
from ....._models import BaseModel
class MessageCreationStepDetails(BaseModel):
    message_creation: MessageCreation
    type: Literal['message_creation']
    'Always `message_creation`.'