import numbers
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence  # noqa
from unittest.mock import Mock
from celery import Celery  # noqa
from celery.canvas import Signature  # noqa
def TaskMessage1(name, id=None, args=(), kwargs=None, callbacks=None, errbacks=None, chain=None, **options):
    """Create task message in protocol 1 format."""
    kwargs = {} if not kwargs else kwargs
    from kombu.serialization import dumps
    from celery import uuid
    id = id or uuid()
    message = Mock(name=f'TaskMessage-{id}')
    message.headers = {}
    message.payload = {'task': name, 'id': id, 'args': args, 'kwargs': kwargs, 'callbacks': callbacks, 'errbacks': errbacks}
    message.payload.update(options)
    message.content_type, message.content_encoding, message.body = dumps(message.payload)
    return message