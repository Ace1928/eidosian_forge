from typing import Optional
from typing_extensions import Literal
from ...._models import BaseModel
class FineTuningJobCheckpoint(BaseModel):
    id: str
    'The checkpoint identifier, which can be referenced in the API endpoints.'
    created_at: int
    'The Unix timestamp (in seconds) for when the checkpoint was created.'
    fine_tuned_model_checkpoint: str
    'The name of the fine-tuned checkpoint model that is created.'
    fine_tuning_job_id: str
    'The name of the fine-tuning job that this checkpoint was created from.'
    metrics: Metrics
    'Metrics at the step number during the fine-tuning job.'
    object: Literal['fine_tuning.job.checkpoint']
    'The object type, which is always "fine_tuning.job.checkpoint".'
    step_number: int
    'The step number that the checkpoint was created at.'