from enum import Enum
from typing import Any, Dict, Optional
from ray._private.pydantic_compat import BaseModel, Field, PYDANTIC_INSTALLED
from ray.dashboard.modules.job.common import JobStatus
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
class JobDetails(BaseModel):
    """
        Job data with extra details about its driver and its submission.
        """
    type: JobType = Field(..., description='The type of job.')
    job_id: Optional[str] = Field(None, description='The job ID. An ID that is created for every job that is launched in Ray. This can be used to fetch data about jobs using Ray Core APIs.')
    submission_id: Optional[str] = Field(None, description='A submission ID is an ID created for every job submitted viathe Ray Jobs API. It can be used to fetch data about jobs using the Ray Jobs API.')
    driver_info: Optional[DriverInfo] = Field(None, description='The driver related to this job. For jobs submitted via the Ray Jobs API, it is the last driver launched by that job submission, or None if there is no driver.')
    status: JobStatus = Field(..., description='The status of the job.')
    entrypoint: str = Field(..., description='The entrypoint command for this job.')
    message: Optional[str] = Field(None, description='A message describing the status in more detail.')
    error_type: Optional[str] = Field(None, description='Internal error or user script error.')
    start_time: Optional[int] = Field(None, description='The time when the job was started. A Unix timestamp in ms.')
    end_time: Optional[int] = Field(None, description='The time when the job moved into a terminal state. A Unix timestamp in ms.')
    metadata: Optional[Dict[str, str]] = Field(None, description='Arbitrary user-provided metadata for the job.')
    runtime_env: Optional[Dict[str, Any]] = Field(None, description='The runtime environment for the job.')
    driver_agent_http_address: Optional[str] = Field(None, description='The HTTP address of the JobAgent on the node the job entrypoint command is running on.')
    driver_node_id: Optional[str] = Field(None, description='The ID of the node the job entrypoint command is running on.')
    driver_exit_code: Optional[int] = Field(None, description="The driver process exit code after the driver executed. Return None if driver doesn't finish executing.")