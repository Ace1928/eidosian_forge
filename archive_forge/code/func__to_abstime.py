from .meta import MetaMessage
def _to_abstime(messages, skip_checks=False):
    """Convert messages to absolute time."""
    now = 0
    for msg in messages:
        now += msg.time
        yield msg.copy(skip_checks=skip_checks, time=now)