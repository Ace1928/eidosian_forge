from typing_extensions import Literal
from ..._models import BaseModel
from .fine_tuning_job_wandb_integration import FineTuningJobWandbIntegration
class FineTuningJobWandbIntegrationObject(BaseModel):
    type: Literal['wandb']
    'The type of the integration being enabled for the fine-tuning job'
    wandb: FineTuningJobWandbIntegration
    'The settings for your integration with Weights and Biases.\n\n    This payload specifies the project that metrics will be sent to. Optionally, you\n    can set an explicit display name for your run, add tags to your run, and set a\n    default entity (team, username, etc) to be associated with your run.\n    '