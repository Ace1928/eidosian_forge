import sys
from functools import partial
import click
from celery.bin.base import LOG_LEVEL, CeleryDaemonCommand, CeleryOption, handle_preload_options
from celery.platforms import detached, set_process_title, strargv
def _run_evtop(app):
    try:
        from celery.events.cursesmon import evtop
        _set_process_status('top')
        return evtop(app=app)
    except ModuleNotFoundError as e:
        if e.name == '_curses':
            raise click.UsageError('The curses module is required for this command.')