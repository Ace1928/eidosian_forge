from __future__ import absolute_import, division, print_function
class GenericRestClient(object):

    def __init__(self, credential, subscription_id, base_url=None, credential_scopes=None):
        self.config = GenericRestClientConfiguration(credential, subscription_id, credential_scopes[0])
        self._client = PipelineClient(base_url, config=self.config)
        self.models = None

    def query(self, url, method, query_parameters, header_parameters, body, expected_status_codes, polling_timeout, polling_interval):
        operation_config = {}
        request = None
        if header_parameters is None:
            header_parameters = {}
        header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if method == 'GET':
            request = self._client.get(url, query_parameters, header_parameters, body)
        elif method == 'PUT':
            request = self._client.put(url, query_parameters, header_parameters, body)
        elif method == 'POST':
            request = self._client.post(url, query_parameters, header_parameters, body)
        elif method == 'HEAD':
            request = self._client.head(url, query_parameters, header_parameters, body)
        elif method == 'PATCH':
            request = self._client.patch(url, query_parameters, header_parameters, body)
        elif method == 'DELETE':
            request = self._client.delete(url, query_parameters, header_parameters, body)
        elif method == 'MERGE':
            request = self._client.merge(url, query_parameters, header_parameters, body)
        response = self._client.send_request(request, **operation_config)
        if response.status_code not in expected_status_codes:
            exp = SendRequestException(response, response.status_code)
            raise exp
        elif response.status_code == 202 and polling_timeout > 0:

            def get_long_running_output(response):
                return response
            poller = LROPoller(self._client, PipelineResponse(None, response, None), get_long_running_output, ARMPolling(polling_interval, **operation_config))
            response = self.get_poller_result(poller, polling_timeout)
        return response

    def get_poller_result(self, poller, timeout):
        try:
            poller.wait(timeout=timeout)
            return poller.result()
        except Exception as exc:
            raise