import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
def _deregister_app(app):
    _apps.discard(app)