import logging
import threading
import urllib
import warnings
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import requests
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
from ray.dashboard.utils import (
from ray.util.annotations import DeveloperAPI
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException, ServerUnavailable
def _make_http_get_request(self, endpoint: str, params: Dict, timeout: float, _explain: bool=False) -> Dict:
    with warnings_on_slow_request(address=self._address, endpoint=endpoint, timeout=timeout, explain=_explain):
        response = None
        try:
            response = self._do_request('GET', endpoint, timeout=timeout, params=params)
            if response.status_code == 500 and 'application/json' not in response.headers.get('Content-Type', ''):
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            err_str = f'Failed to make request to {self._address}{endpoint}. '
            err_str += 'Failed to connect to API server. Please check the API server log for details. Make sure dependencies are installed with `pip install ray[default]`. Please also check dashboard is available, and included when starting ray cluster, i.e. `ray start --include-dashboard=True --head`. '
            if response is None:
                raise ServerUnavailable(err_str)
            err_str += f'Response(url={response.url},status={response.status_code})'
            raise RayStateApiException(err_str) from e
    response = response.json()
    if response['result'] is False:
        raise RayStateApiException(f'API server internal error. See dashboard.log file for more details. Error: {response['msg']}')
    return response['data']['result']