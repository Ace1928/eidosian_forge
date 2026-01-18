class DelayedPretty(object):
    """Wraps a message and delays prettifying it until requested.

    TODO(harlowja): remove this when https://github.com/celery/kombu/pull/454/
    is merged and a release is made that contains it (since that pull
    request is equivalent and/or better than this).
    """

    def __init__(self, message):
        self._message = message
        self._message_pretty = None

    def __str__(self):
        if self._message_pretty is None:
            self._message_pretty = _prettify_message(self._message)
        return self._message_pretty