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
@DeveloperAPI
class StateApiClient(SubmissionClient):
    """State API Client issues REST GET requests to the server for resource states."""

    def __init__(self, address: Optional[str]=None, cookies: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, Any]]=None):
        """Initialize a StateApiClient and check the connection to the cluster.

        Args:
            address: Ray bootstrap address (e.g. `127.0.0.0:6379`, `auto`), or Ray
                Client adress (e.g. `ray://<head-node-ip>:10001`), or Ray dashboard
                address (e.g. `http://<head-node-ip>:8265`).
                If not provided, it will be detected automatically from any running
                local Ray cluster.
            cookies: Cookies to use when sending requests to the HTTP job server.
            headers: Headers to use when sending requests to the HTTP job server, used
                for cases like authentication to a remote cluster.
        """
        if requests is None:
            raise RuntimeError("The Ray state CLI & SDK require the ray[default] installation: `pip install 'ray[default']``")
        if not headers:
            headers = {'Content-Type': 'application/json'}
        api_server_url = get_address_for_submission_client(address)
        super().__init__(address=api_server_url, create_cluster_if_needed=False, headers=headers, cookies=cookies)

    @classmethod
    def _make_param(cls, options: Union[ListApiOptions, GetApiOptions]) -> Dict:
        options_dict = {}
        for field in fields(options):
            if field.name == 'filters':
                options_dict['filter_keys'] = []
                options_dict['filter_predicates'] = []
                options_dict['filter_values'] = []
                for filter in options.filters:
                    if len(filter) != 3:
                        raise ValueError(f'The given filter has incorrect input type, {filter}. Provide (key, predicate, value) tuples.')
                    filter_k, filter_predicate, filter_val = filter
                    options_dict['filter_keys'].append(filter_k)
                    options_dict['filter_predicates'].append(filter_predicate)
                    options_dict['filter_values'].append(filter_val)
                continue
            option_val = getattr(options, field.name)
            if option_val is not None:
                options_dict[field.name] = option_val
        return options_dict

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

    def get(self, resource: StateResource, id: str, options: Optional[GetApiOptions], _explain: bool=False) -> Optional[Union[ActorState, PlacementGroupState, NodeState, WorkerState, TaskState, List[ObjectState], JobState]]:
        """Get resources states by id

        Args:
            resource_name: Resource names, i.e. 'workers', 'actors', 'nodes',
                'placement_groups', 'tasks', 'objects'.
                'jobs' and 'runtime-envs' are not supported yet.
            id: ID for the resource, i.e. 'node_id' for nodes.
            options: Get options. See `GetApiOptions` for details.
            _explain: Print the API information such as API
                latency or failed query information.

        Returns:
            None if not found, and if found:
            - ActorState for actors
            - PlacementGroupState for placement groups
            - NodeState for nodes
            - WorkerState for workers
            - TaskState for tasks
            - JobState for jobs

            Empty list for objects if not found, or list of ObjectState for objects

        Raises:
            This doesn't catch any exceptions raised when the underlying request
            call raises exceptions. For example, it could raise `requests.Timeout`
            when timeout occurs.

            ValueError:
                if the resource could not be GET by id, i.e. jobs and runtime-envs.

        """
        params = self._make_param(options)
        RESOURCE_ID_KEY_NAME = {StateResource.NODES: 'node_id', StateResource.ACTORS: 'actor_id', StateResource.PLACEMENT_GROUPS: 'placement_group_id', StateResource.WORKERS: 'worker_id', StateResource.TASKS: 'task_id', StateResource.OBJECTS: 'object_id', StateResource.JOBS: 'submission_id'}
        if resource not in RESOURCE_ID_KEY_NAME:
            raise ValueError(f"Can't get {resource.name} by id.")
        params['filter_keys'] = [RESOURCE_ID_KEY_NAME[resource]]
        params['filter_predicates'] = ['=']
        params['filter_values'] = [id]
        params['detail'] = True
        endpoint = f'/api/v0/{resource.value}'
        list_api_response = self._make_http_get_request(endpoint=endpoint, params=params, timeout=options.timeout, _explain=_explain)
        result = list_api_response['result']
        if len(result) == 0:
            return None
        result = [dict_to_state(d, resource) for d in result]
        if resource == StateResource.OBJECTS:
            return result
        if resource == StateResource.TASKS:
            if len(result) == 1:
                return result[0]
            return result
        assert len(result) == 1
        return result[0]

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

    def _raise_on_missing_output(self, resource: StateResource, api_response: dict):
        """Raise an exception when the API resopnse contains a missing output.

        Output can be missing if (1) Failures on some of data source queries (e.g.,
        `ray list tasks` queries all raylets, and if some of queries fail, it will
        contain missing output. If all queries fail, it will just fail). (2) Data
        is truncated because the output is too large.

        Args:
            resource: Resource names, i.e. 'jobs', 'actors', 'nodes',
                see `StateResource` for details.
            api_response: The dictionarified `ListApiResponse` or `SummaryApiResponse`.
        """
        warning_msgs = api_response.get('partial_failure_warning', None)
        if warning_msgs:
            raise RayStateApiException(f'Failed to retrieve all {resource.value} from the cluster becausethey are not reachable due to query failures to the data sources. To avoid raising an exception and allow having missing output, set `raise_on_missing_output=False`. ')
        total = api_response['total']
        num_after_truncation = api_response['num_after_truncation']
        if total != num_after_truncation:
            raise RayStateApiException(f'Failed to retrieve all {total} {resource.value} from the cluster because they are not reachable due to data truncation. It happens when the returned data is too large (> {num_after_truncation}) To avoid raising an exception and allow having missing output, set `raise_on_missing_output=False`. ')

    def list(self, resource: StateResource, options: ListApiOptions, raise_on_missing_output: bool, _explain: bool=False) -> List[Union[ActorState, JobState, NodeState, TaskState, ObjectState, PlacementGroupState, RuntimeEnvState, WorkerState, ClusterEventState]]:
        """List resources states

        Args:
            resource: Resource names, i.e. 'jobs', 'actors', 'nodes',
                see `StateResource` for details.
            options: List options. See `ListApiOptions` for details.
            raise_on_missing_output: When True, raise an exception if the output
                is incomplete. Output can be incomplete if
                (1) there's a partial network failure when the source is distributed.
                (2) data is truncated because it is too large.
                Set it to False to avoid throwing an exception on missing data.
            _explain: Print the API information such as API
                latency or failed query information.

        Returns:
            A list of queried result from `ListApiResponse`,

        Raises:
            This doesn't catch any exceptions raised when the underlying request
            call raises exceptions. For example, it could raise `requests.Timeout`
            when timeout occurs.

        """
        endpoint = f'/api/v0/{resource.value}'
        params = self._make_param(options)
        list_api_response = self._make_http_get_request(endpoint=endpoint, params=params, timeout=options.timeout, _explain=_explain)
        if raise_on_missing_output:
            self._raise_on_missing_output(resource, list_api_response)
        if _explain:
            self._print_api_warning(resource, list_api_response)
        return [dict_to_state(d, resource) for d in list_api_response['result']]

    def summary(self, resource: SummaryResource, *, options: SummaryApiOptions, raise_on_missing_output: bool, _explain: bool=False) -> Dict:
        """Summarize resources states

        Args:
            resource_name: Resource names,
                see `SummaryResource` for details.
            options: summary options. See `SummaryApiOptions` for details.
            raise_on_missing_output: Raise an exception if the output has missing data.
                Output can have missing data if (1) there's a partial network failure
                when the source is distributed. (2) data is truncated
                because it is too large.
            _explain: Print the API information such as API
                latency or failed query information.

        Returns:
            A dictionary of queried result from `SummaryApiResponse`.

        Raises:
            This doesn't catch any exceptions raised when the underlying request
            call raises exceptions. For example, it could raise `requests.Timeout`
            when timeout occurs.
        """
        params = {'timeout': options.timeout}
        endpoint = f'/api/v0/{resource.value}/summarize'
        summary_api_response = self._make_http_get_request(endpoint=endpoint, params=params, timeout=options.timeout, _explain=_explain)
        if raise_on_missing_output:
            self._raise_on_missing_output(resource, summary_api_response)
        if _explain:
            self._print_api_warning(resource, summary_api_response, warn_limit=False)
        return summary_api_response['result']['node_id_to_summary']