from typing_extensions import Literal
from ..._models import BaseModel
class FineTuningJobEvent(BaseModel):
    id: str
    created_at: int
    level: Literal['info', 'warn', 'error']
    message: str
    object: Literal['fine_tuning.job.event']