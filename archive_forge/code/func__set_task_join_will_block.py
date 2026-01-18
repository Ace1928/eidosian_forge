import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
def _set_task_join_will_block(blocks):
    global _task_join_will_block
    _task_join_will_block = blocks