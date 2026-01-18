from __future__ import absolute_import
import ast
from sentry_sdk import Hub, serializer
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.utils import walk_exception_chain, iter_stacks
class PureEvalIntegration(Integration):
    identifier = 'pure_eval'

    @staticmethod
    def setup_once():

        @add_global_event_processor
        def add_executing_info(event, hint):
            if Hub.current.get_integration(PureEvalIntegration) is None:
                return event
            if hint is None:
                return event
            exc_info = hint.get('exc_info', None)
            if exc_info is None:
                return event
            exception = event.get('exception', None)
            if exception is None:
                return event
            values = exception.get('values', None)
            if values is None:
                return event
            for exception, (_exc_type, _exc_value, exc_tb) in zip(reversed(values), walk_exception_chain(exc_info)):
                sentry_frames = [frame for frame in exception.get('stacktrace', {}).get('frames', []) if frame.get('function')]
                tbs = list(iter_stacks(exc_tb))
                if len(sentry_frames) != len(tbs):
                    continue
                for sentry_frame, tb in zip(sentry_frames, tbs):
                    sentry_frame['vars'] = pure_eval_frame(tb.tb_frame) or sentry_frame['vars']
            return event