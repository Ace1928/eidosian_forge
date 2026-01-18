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
def _print_api_warning(self, resource: StateResource, api_response: dict, warn_data_source_not_available: bool=True, warn_data_truncation: bool=True, warn_limit: bool=True, warn_server_side_warnings: bool=True):
    """Print the API warnings.

        Args:
            resource: Resource names, i.e. 'jobs', 'actors', 'nodes',
                see `StateResource` for details.
            api_response: The dictionarified `ListApiResponse` or `SummaryApiResponse`.
            warn_data_source_not_available: Warn when some data sources
                are not available.
            warn_data_truncation: Warn when results were truncated at
                the data source.
            warn_limit: Warn when results were limited.
            warn_server_side_warnings: Warn when the server side generates warnings
                (E.g., when callsites not enabled for listing objects)
        """
    if warn_data_source_not_available:
        warning_msgs = api_response.get('partial_failure_warning', None)
        if warning_msgs:
            warnings.warn(warning_msgs)
    if warn_data_truncation:
        num_after_truncation = api_response['num_after_truncation']
        total = api_response['total']
        if total > num_after_truncation:
            warnings.warn(f'The returned data may contain incomplete result. {num_after_truncation} ({total} total from the cluster) {resource.value} are retrieved from the data source. {total - num_after_truncation} entries have been truncated. Max of {num_after_truncation} entries are retrieved from data source to prevent over-sized payloads.')
    if warn_limit:
        num_filtered = api_response['num_filtered']
        data = api_response['result']
        if num_filtered > len(data):
            warnings.warn(f'Limit last {len(data)} entries (Total {num_filtered}). Use `--filter` to reduce the amount of data to return or setting a higher limit with `--limit` to see all data. ')
    if warn_server_side_warnings:
        warnings_to_print = api_response.get('warnings', [])
        if warnings_to_print:
            for warning_to_print in warnings_to_print:
                warnings.warn(warning_to_print)