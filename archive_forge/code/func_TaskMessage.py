import numbers
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence  # noqa
from unittest.mock import Mock
from celery import Celery  # noqa
from celery.canvas import Signature  # noqa
def TaskMessage(name, id=None, args=(), kwargs=None, callbacks=None, errbacks=None, chain=None, shadow=None, utc=None, **options):
    """Create task message in protocol 2 format."""
    kwargs = {} if not kwargs else kwargs
    from kombu.serialization import dumps
    from celery import uuid
    id = id or uuid()
    message = Mock(name=f'TaskMessage-{id}')
    message.headers = {'id': id, 'task': name, 'shadow': shadow}
    embed = {'callbacks': callbacks, 'errbacks': errbacks, 'chain': chain}
    message.headers.update(options)
    message.content_type, message.content_encoding, message.body = dumps((args, kwargs, embed), serializer='json')
    message.payload = (args, kwargs, embed)
    return message