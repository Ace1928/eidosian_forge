from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
class BuilderConfig(BaseModel):
    type: Optional[BuilderType] = Field(None, description='The type of builder to use.')
    destination: Optional[str] = Field(None, description='The destination to use for the built image. If not provided, the image will be pushed to the registry.')
    platform: Optional[TargetPlatform] = Field(None, description='The platform to use for the built image. If not provided, the platform will be detected automatically.')
    build_context_store: Optional[str] = Field(None, description='The build context store to use. Required for kaniko builds.', alias='build-context-store')
    build_job_name: Optional[str] = Field('wandb-launch-container-build', description='Name prefix of the build job.', alias='build-job-name')
    secret_name: Optional[str] = Field(None, description='The name of the secret to use for the build job.', alias='secret-name')
    secret_key: Optional[str] = Field(None, description='The key of the secret to use for the build job.', alias='secret-key')
    kaniko_image: Optional[str] = Field('gcr.io/kaniko-project/executor:latest', description='The image to use for the kaniko executor.', alias='kaniko-image')

    @validator('build_context_store')
    @classmethod
    def validate_build_context_store(cls, build_context_store: Optional[str]) -> Optional[str]:
        """Validate that the build context store is a valid container registry URI."""
        if build_context_store is None:
            return None
        for regex in [S3_URI_RE, GCS_URI_RE, AZURE_BLOB_REGEX]:
            if regex.match(build_context_store):
                return build_context_store
        raise ValueError('Invalid build context store. Build context store must be a URI for an S3 bucket, GCS bucket, or Azure blob.')

    @root_validator(pre=True)
    @classmethod
    def validate_docker(cls, values: dict) -> dict:
        """Right now there are no required fields for docker builds."""
        return values