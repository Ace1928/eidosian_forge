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
class _GetConfig(SpeechRestStub):

    def __hash__(self):
        return hash('GetConfig')
    __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

    @classmethod
    def _get_unset_required_fields(cls, message_dict):
        return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

    def __call__(self, request: cloud_speech.GetConfigRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> cloud_speech.Config:
        """Call the get config method over HTTP.

            Args:
                request (~.cloud_speech.GetConfigRequest):
                    The request object. Request message for the
                [GetConfig][google.cloud.speech.v2.Speech.GetConfig]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.cloud_speech.Config:
                    Message representing the config for the Speech-to-Text
                API. This includes an optional `KMS
                key <https://cloud.google.com/kms/docs/resource-hierarchy#keys>`__
                with which incoming data will be encrypted.

            """
        http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v2/{name=projects/*/locations/*/config}'}]
        request, metadata = self._interceptor.pre_get_config(request, metadata)
        pb_request = cloud_speech.GetConfigRequest.pb(request)
        transcoded_request = path_template.transcode(http_options, pb_request)
        uri = transcoded_request['uri']
        method = transcoded_request['method']
        query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
        query_params.update(self._get_unset_required_fields(query_params))
        query_params['$alt'] = 'json;enum-encoding=int'
        headers = dict(metadata)
        headers['Content-Type'] = 'application/json'
        response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)
        resp = cloud_speech.Config()
        pb_resp = cloud_speech.Config.pb(resp)
        json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
        resp = self._interceptor.post_get_config(resp)
        return resp