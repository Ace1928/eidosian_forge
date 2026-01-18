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
@retry(retry=retry_if_exception_type(TimeoutError), stop=stop_after_attempt(2), reraise=True)
def _rpcq_request(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
    engagement = self._engagement_manager.get_engagement(endpoint_id=self._endpoint_id, quantum_processor_id=self.quantum_processor_id, request_timeout=self.timeout)
    client = rpcq.Client(endpoint=engagement.address, timeout=self._calculate_timeout(engagement), auth_config=self._auth_config(engagement.credentials))
    try:
        return client.call(method_name, *args, **kwargs)
    except TimeoutError as e:
        raise TimeoutError(f'Request to QPU at {engagement.address} timed out. See the Troubleshooting Guide: {DOCS_URL}/troubleshooting.html') from e
    finally:
        client.close()