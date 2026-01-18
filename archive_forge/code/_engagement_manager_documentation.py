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

        Gets an engagement for the given quantum processor endpoint.

        If an engagement was already fetched previously and remains valid, it will be returned instead
        of creating a new engagement.

        :param quantum_processor_id: Quantum processor being engaged.
        :param request_timeout: Timeout for request, in seconds.
        :param endpoint_id: Optional ID of the endpoint to use for engagement. If provided, it must
            correspond to an endpoint serving the provided Quantum Processor.
        :return: Fetched or cached engagement.
        :raises QPUUnavailableError: raised when the QPU is unavailable due, and provides a suggested
            number of seconds to wait until retrying.
        :raises QCSHTTPStatusError: raised when creating an engagement fails for a reason that is not
            due to QPU unavailability.
        