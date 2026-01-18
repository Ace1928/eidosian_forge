import hashlib
from functools import cached_property
from inspect import isawaitable
from sentry_sdk import configure_scope, start_span
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class StrawberryIntegration(Integration):
    identifier = 'strawberry'

    def __init__(self, async_execution=None):
        if async_execution not in (None, False, True):
            raise ValueError('Invalid value for async_execution: "{}" (must be bool)'.format(async_execution))
        self.async_execution = async_execution

    @staticmethod
    def setup_once():
        version = package_version('strawberry-graphql')
        if version is None:
            raise DidNotEnable('Unparsable strawberry-graphql version: {}'.format(version))
        if version < (0, 209, 5):
            raise DidNotEnable('strawberry-graphql 0.209.5 or newer required.')
        _patch_schema_init()
        _patch_execute()
        _patch_views()