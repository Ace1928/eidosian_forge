import dataclasses
import json  # type: ignore
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.api_core import (
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.speech_v2.types import cloud_speech
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import SpeechTransport
class _UpdateRecognizer(SpeechRestStub):

    def __hash__(self):
        return hash('UpdateRecognizer')
    __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

    @classmethod
    def _get_unset_required_fields(cls, message_dict):
        return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

    def __call__(self, request: cloud_speech.UpdateRecognizerRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> operations_pb2.Operation:
        """Call the update recognizer method over HTTP.

            Args:
                request (~.cloud_speech.UpdateRecognizerRequest):
                    The request object. Request message for the
                [UpdateRecognizer][google.cloud.speech.v2.Speech.UpdateRecognizer]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """
        http_options: List[Dict[str, str]] = [{'method': 'patch', 'uri': '/v2/{recognizer.name=projects/*/locations/*/recognizers/*}', 'body': 'recognizer'}]
        request, metadata = self._interceptor.pre_update_recognizer(request, metadata)
        pb_request = cloud_speech.UpdateRecognizerRequest.pb(request)
        transcoded_request = path_template.transcode(http_options, pb_request)
        body = json_format.MessageToJson(transcoded_request['body'], use_integers_for_enums=True)
        uri = transcoded_request['uri']
        method = transcoded_request['method']
        query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
        query_params.update(self._get_unset_required_fields(query_params))
        query_params['$alt'] = 'json;enum-encoding=int'
        headers = dict(metadata)
        headers['Content-Type'] = 'application/json'
        response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)
        resp = operations_pb2.Operation()
        json_format.Parse(response.content, resp, ignore_unknown_fields=True)
        resp = self._interceptor.post_update_recognizer(resp)
        return resp