import os
import subprocess
import sys
import platform
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.tracing_utils import EnvironHeaders, should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class StdlibIntegration(Integration):
    identifier = 'stdlib'

    @staticmethod
    def setup_once():
        _install_httplib()
        _install_subprocess()

        @add_global_event_processor
        def add_python_runtime_context(event, hint):
            if Hub.current.get_integration(StdlibIntegration) is not None:
                contexts = event.setdefault('contexts', {})
                if isinstance(contexts, dict) and 'runtime' not in contexts:
                    contexts['runtime'] = _RUNTIME_CONTEXT
            return event