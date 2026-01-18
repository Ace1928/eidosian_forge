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
from google.pubsub_v1.types import pubsub
from .base import SubscriberTransport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class _UpdateSubscription(SubscriberRestStub):

    def __hash__(self):
        return hash('UpdateSubscription')
    __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

    @classmethod
    def _get_unset_required_fields(cls, message_dict):
        return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

    def __call__(self, request: pubsub.UpdateSubscriptionRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.Subscription:
        """Call the update subscription method over HTTP.

            Args:
                request (~.pubsub.UpdateSubscriptionRequest):
                    The request object. Request for the UpdateSubscription
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.Subscription:
                    A subscription resource. If none of ``push_config``,
                ``bigquery_config``, or ``cloud_storage_config`` is set,
                then the subscriber will pull and ack messages using API
                methods. At most one of these fields may be set.

            """
        http_options: List[Dict[str, str]] = [{'method': 'patch', 'uri': '/v1/{subscription.name=projects/*/subscriptions/*}', 'body': '*'}]
        request, metadata = self._interceptor.pre_update_subscription(request, metadata)
        pb_request = pubsub.UpdateSubscriptionRequest.pb(request)
        transcoded_request = path_template.transcode(http_options, pb_request)
        body = json_format.MessageToJson(transcoded_request['body'], including_default_value_fields=False, use_integers_for_enums=True)
        uri = transcoded_request['uri']
        method = transcoded_request['method']
        query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
        query_params.update(self._get_unset_required_fields(query_params))
        query_params['$alt'] = 'json;enum-encoding=int'
        headers = dict(metadata)
        headers['Content-Type'] = 'application/json'
        response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)
        resp = pubsub.Subscription()
        pb_resp = pubsub.Subscription.pb(resp)
        json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
        resp = self._interceptor.post_update_subscription(resp)
        return resp