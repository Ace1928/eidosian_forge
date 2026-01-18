import sys
from functools import partial
import click
from celery.bin.base import LOG_LEVEL, CeleryDaemonCommand, CeleryOption, handle_preload_options
from celery.platforms import detached, set_process_title, strargv
def _run_evdump(app):
    from celery.events.dumper import evdump
    _set_process_status('dump')
    return evdump(app=app)