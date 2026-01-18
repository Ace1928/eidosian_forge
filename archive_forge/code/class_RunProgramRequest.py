from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, cast, Tuple, Union, List, Any
from attr import field
import rpcq
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from pyquil.api import EngagementManager
from pyquil._version import DOCS_URL
@dataclass
class RunProgramRequest:
    """
    Request to run a program.
    """
    id: str
    'Identifier for request.'
    priority: int
    'Priority for request.'
    program: str
    'Encrypted program to run.'
    patch_values: Dict[str, List[Union[int, float]]]
    'Map of data names to data values for patching the program.'