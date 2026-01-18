from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
class AgentConfig(BaseModel):
    """Configuration for the Launch agent."""
    queues: List[str] = Field(default=[], description='The queues to use for this agent.')
    project: Optional[str] = Field(description='The W&B project to use for this agent.')
    entity: Optional[str] = Field(description='The W&B entity to use for this agent.')
    max_jobs: Optional[int] = Field(1, description='The maximum number of jobs to run concurrently.')
    max_schedulers: Optional[int] = Field(1, description='The maximum number of sweep schedulers to run concurrently.')
    secure_mode: Optional[bool] = Field(False, description='Whether to use secure mode for this agent. If True, the agent will reject runs that attempt to override the entrypoint or image.')
    registry: Optional[RegistryConfig] = Field(None, description='The registry to use.')
    environment: Optional[EnvironmentConfig] = Field(None, description='The environment to use.')
    builder: Optional[BuilderConfig] = Field(None, description='The builder to use.')

    class Config:
        extra = 'forbid'