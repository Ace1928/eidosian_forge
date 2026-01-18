import threading
from datetime import datetime
from typing import Dict, NamedTuple, Optional, TYPE_CHECKING
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models import EngagementWithCredentials, CreateEngagementRequest
from qcs_api_client.operations.sync import create_engagement
from qcs_api_client.types import UNSET
from qcs_api_client.util.errors import QCSHTTPStatusError
from pyquil.api._qcs_client import qcs_client
@staticmethod
def _engagement_valid(engagement: Optional[EngagementWithCredentials]) -> bool:
    if engagement is None:
        return False
    return all([engagement.credentials.client_public != '', engagement.credentials.client_secret != '', engagement.credentials.server_public != '', parsedate(engagement.expires_at) > datetime.now(tzutc()), engagement.address != ''])