import sys
from functools import partial
import click
from celery.bin.base import LOG_LEVEL, CeleryDaemonCommand, CeleryOption, handle_preload_options
from celery.platforms import detached, set_process_title, strargv
def _run_evcam(camera, app, logfile=None, pidfile=None, uid=None, gid=None, umask=None, workdir=None, detach=False, **kwargs):
    from celery.events.snapshot import evcam
    _set_process_status('cam')
    kwargs['app'] = app
    cam = partial(evcam, camera, logfile=logfile, pidfile=pidfile, **kwargs)
    if detach:
        with detached(logfile, pidfile, uid, gid, umask, workdir):
            return cam()
    else:
        return cam()