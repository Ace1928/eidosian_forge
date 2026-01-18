from typing import List, Union, Optional
from typing_extensions import Literal
from ..._models import BaseModel
class FineTuningJob(BaseModel):
    id: str
    'The object identifier, which can be referenced in the API endpoints.'
    created_at: int
    'The Unix timestamp (in seconds) for when the fine-tuning job was created.'
    error: Optional[Error] = None
    '\n    For fine-tuning jobs that have `failed`, this will contain more information on\n    the cause of the failure.\n    '
    fine_tuned_model: Optional[str] = None
    'The name of the fine-tuned model that is being created.\n\n    The value will be null if the fine-tuning job is still running.\n    '
    finished_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the fine-tuning job was finished.\n\n    The value will be null if the fine-tuning job is still running.\n    '
    hyperparameters: Hyperparameters
    'The hyperparameters used for the fine-tuning job.\n\n    See the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)\n    for more details.\n    '
    model: str
    'The base model that is being fine-tuned.'
    object: Literal['fine_tuning.job']
    'The object type, which is always "fine_tuning.job".'
    organization_id: str
    'The organization that owns the fine-tuning job.'
    result_files: List[str]
    'The compiled results file ID(s) for the fine-tuning job.\n\n    You can retrieve the results with the\n    [Files API](https://platform.openai.com/docs/api-reference/files/retrieve-contents).\n    '
    status: Literal['validating_files', 'queued', 'running', 'succeeded', 'failed', 'cancelled']
    '\n    The current status of the fine-tuning job, which can be either\n    `validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.\n    '
    trained_tokens: Optional[int] = None
    'The total number of billable tokens processed by this fine-tuning job.\n\n    The value will be null if the fine-tuning job is still running.\n    '
    training_file: str
    'The file ID used for training.\n\n    You can retrieve the training data with the\n    [Files API](https://platform.openai.com/docs/api-reference/files/retrieve-contents).\n    '
    validation_file: Optional[str] = None
    'The file ID used for validation.\n\n    You can retrieve the validation results with the\n    [Files API](https://platform.openai.com/docs/api-reference/files/retrieve-contents).\n    '