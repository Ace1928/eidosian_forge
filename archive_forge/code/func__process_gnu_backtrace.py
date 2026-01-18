import re
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def _process_gnu_backtrace(event, hint):
    if Hub.current.get_integration(GnuBacktraceIntegration) is None:
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
    for exception in values:
        frames = exception.get('stacktrace', {}).get('frames', [])
        if not frames:
            continue
        msg = exception.get('value', None)
        if not msg:
            continue
        additional_frames = []
        new_msg = []
        for line in msg.splitlines():
            match = FRAME_RE.match(line)
            if match:
                additional_frames.append((int(match.group('index')), {'package': match.group('package') or None, 'function': match.group('function') or None, 'platform': 'native'}))
            else:
                new_msg.append(line)
        if additional_frames:
            additional_frames.sort(key=lambda x: -x[0])
            for _, frame in additional_frames:
                frames.append(frame)
            new_msg.append('<stacktrace parsed and removed by GnuBacktraceIntegration>')
            exception['value'] = '\n'.join(new_msg)
    return event