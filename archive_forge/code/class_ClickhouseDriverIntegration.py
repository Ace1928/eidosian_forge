from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import capture_internal_exceptions
from typing import TypeVar
class ClickhouseDriverIntegration(Integration):
    identifier = 'clickhouse_driver'

    @staticmethod
    def setup_once() -> None:
        clickhouse_driver.connection.Connection.send_query = _wrap_start(clickhouse_driver.connection.Connection.send_query)
        clickhouse_driver.client.Client.send_data = _wrap_send_data(clickhouse_driver.client.Client.send_data)
        clickhouse_driver.client.Client.receive_end_of_query = _wrap_end(clickhouse_driver.client.Client.receive_end_of_query)
        if hasattr(clickhouse_driver.client.Client, 'receive_end_of_insert_query'):
            clickhouse_driver.client.Client.receive_end_of_insert_query = _wrap_end(clickhouse_driver.client.Client.receive_end_of_insert_query)
        clickhouse_driver.client.Client.receive_result = _wrap_end(clickhouse_driver.client.Client.receive_result)