from google.auth.transport.requests import AuthorizedSession  # type: ignore
import json  # type: ignore
import grpc  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import path_template
from google.api_core import gapic_v1
from cloudsdk.google.protobuf import json_format
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from google.pubsub_v1.types import schema
from google.pubsub_v1.types import schema as gp_schema
from .base import (
class _ListSchemaRevisions(SchemaServiceRestStub):

    def __hash__(self):
        return hash('ListSchemaRevisions')
    __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

    @classmethod
    def _get_unset_required_fields(cls, message_dict):
        return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

    def __call__(self, request: schema.ListSchemaRevisionsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> schema.ListSchemaRevisionsResponse:
        """Call the list schema revisions method over HTTP.

            Args:
                request (~.schema.ListSchemaRevisionsRequest):
                    The request object. Request for the ``ListSchemaRevisions`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.schema.ListSchemaRevisionsResponse:
                    Response for the ``ListSchemaRevisions`` method.
            """
        http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/{name=projects/*/schemas/*}:listRevisions'}]
        request, metadata = self._interceptor.pre_list_schema_revisions(request, metadata)
        pb_request = schema.ListSchemaRevisionsRequest.pb(request)
        transcoded_request = path_template.transcode(http_options, pb_request)
        uri = transcoded_request['uri']
        method = transcoded_request['method']
        query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
        query_params.update(self._get_unset_required_fields(query_params))
        query_params['$alt'] = 'json;enum-encoding=int'
        headers = dict(metadata)
        headers['Content-Type'] = 'application/json'
        response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)
        resp = schema.ListSchemaRevisionsResponse()
        pb_resp = schema.ListSchemaRevisionsResponse.pb(resp)
        json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
        resp = self._interceptor.post_list_schema_revisions(resp)
        return resp