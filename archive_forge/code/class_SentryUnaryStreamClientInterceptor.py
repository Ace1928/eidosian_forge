from typing import Callable, Union, AsyncIterable, Any
from grpc.aio import (
from google.protobuf.message import Message
from sentry_sdk import Hub
from sentry_sdk.consts import OP
class SentryUnaryStreamClientInterceptor(ClientInterceptor, UnaryStreamClientInterceptor):

    async def intercept_unary_stream(self, continuation: Callable[[ClientCallDetails, Message], UnaryStreamCall], client_call_details: ClientCallDetails, request: Message) -> Union[AsyncIterable[Any], UnaryStreamCall]:
        hub = Hub.current
        method = client_call_details.method
        with hub.start_span(op=OP.GRPC_CLIENT, description='unary stream call to %s' % method.decode()) as span:
            span.set_data('type', 'unary stream')
            span.set_data('method', method)
            client_call_details = self._update_client_call_details_metadata_from_hub(client_call_details, hub)
            response = await continuation(client_call_details, request)
            return response