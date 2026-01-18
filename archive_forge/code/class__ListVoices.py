import dataclasses
import json  # type: ignore
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.api_core import gapic_v1, path_template, rest_helpers, rest_streaming
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import TextToSpeechTransport
class _ListVoices(TextToSpeechRestStub):

    def __hash__(self):
        return hash('ListVoices')

    def __call__(self, request: cloud_tts.ListVoicesRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> cloud_tts.ListVoicesResponse:
        """Call the list voices method over HTTP.

            Args:
                request (~.cloud_tts.ListVoicesRequest):
                    The request object. The top-level message sent by the client for the
                ``ListVoices`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.cloud_tts.ListVoicesResponse:
                    The message returned to the client by the ``ListVoices``
                method.

            """
        http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/voices'}]
        request, metadata = self._interceptor.pre_list_voices(request, metadata)
        pb_request = cloud_tts.ListVoicesRequest.pb(request)
        transcoded_request = path_template.transcode(http_options, pb_request)
        uri = transcoded_request['uri']
        method = transcoded_request['method']
        query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=False))
        headers = dict(metadata)
        headers['Content-Type'] = 'application/json'
        response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)
        resp = cloud_tts.ListVoicesResponse()
        pb_resp = cloud_tts.ListVoicesResponse.pb(resp)
        json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
        resp = self._interceptor.post_list_voices(resp)
        return resp