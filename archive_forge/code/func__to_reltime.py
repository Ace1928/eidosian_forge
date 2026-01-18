from .meta import MetaMessage
def _to_reltime(messages, skip_checks=False):
    """Convert messages to relative time."""
    now = 0
    for msg in messages:
        delta = msg.time - now
        yield msg.copy(skip_checks=skip_checks, time=delta)
        now = msg.time