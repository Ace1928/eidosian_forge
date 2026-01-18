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
def _calculate_timeout(self, engagement: EngagementWithCredentials) -> float:
    engagement_time_left = parsedate(engagement.expires_at) - datetime.now(tzutc())
    return min(self.timeout, engagement_time_left.total_seconds())