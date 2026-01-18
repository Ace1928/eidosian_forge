from enum import Enum
from typing import Any, Dict, Optional
from ray._private.pydantic_compat import BaseModel, Field, PYDANTIC_INSTALLED
from ray.dashboard.modules.job.common import JobStatus
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
class DriverInfo(BaseModel):
    """A class for recording information about the driver related to the job."""
    id: str = Field(..., description='The id of the driver')
    node_ip_address: str = Field(..., description='The IP address of the node the driver is running on.')
    pid: str = Field(..., description='The PID of the worker process the driver is using.')