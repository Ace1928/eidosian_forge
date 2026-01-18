from sentry_sdk.utils import event_from_exception, parse_version
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk._types import TYPE_CHECKING
class GQLIntegration(Integration):
    identifier = 'gql'

    @staticmethod
    def setup_once():
        gql_version = parse_version(gql.__version__)
        if gql_version is None or gql_version < MIN_GQL_VERSION:
            raise DidNotEnable('GQLIntegration is only supported for GQL versions %s and above.' % '.'.join((str(num) for num in MIN_GQL_VERSION)))
        _patch_execute()