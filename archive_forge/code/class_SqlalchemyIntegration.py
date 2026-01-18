from __future__ import absolute_import
from sentry_sdk._compat import text_type
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import SPANDATA
from sentry_sdk.db.explain_plan.sqlalchemy import attach_explain_plan_to_span
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import capture_internal_exceptions, parse_version
class SqlalchemyIntegration(Integration):
    identifier = 'sqlalchemy'

    @staticmethod
    def setup_once():
        version = parse_version(SQLALCHEMY_VERSION)
        if version is None:
            raise DidNotEnable('Unparsable SQLAlchemy version: {}'.format(SQLALCHEMY_VERSION))
        if version < (1, 2):
            raise DidNotEnable('SQLAlchemy 1.2 or newer required.')
        listen(Engine, 'before_cursor_execute', _before_cursor_execute)
        listen(Engine, 'after_cursor_execute', _after_cursor_execute)
        listen(Engine, 'handle_error', _handle_error)