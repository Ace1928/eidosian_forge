import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
class WorkerClass(Setting):
    name = 'worker_class'
    section = 'Worker Processes'
    cli = ['-k', '--worker-class']
    meta = 'STRING'
    validator = validate_class
    default = 'sync'
    desc = '        The type of workers to use.\n\n        The default class (``sync``) should handle most "normal" types of\n        workloads. You\'ll want to read :doc:`design` for information on when\n        you might want to choose one of the other worker classes. Required\n        libraries may be installed using setuptools\' ``extras_require`` feature.\n\n        A string referring to one of the following bundled classes:\n\n        * ``sync``\n        * ``eventlet`` - Requires eventlet >= 0.24.1 (or install it via\n          ``pip install gunicorn[eventlet]``)\n        * ``gevent``   - Requires gevent >= 1.4 (or install it via\n          ``pip install gunicorn[gevent]``)\n        * ``tornado``  - Requires tornado >= 0.2 (or install it via\n          ``pip install gunicorn[tornado]``)\n        * ``gthread``  - Python 2 requires the futures package to be installed\n          (or install it via ``pip install gunicorn[gthread]``)\n\n        Optionally, you can provide your own worker by giving Gunicorn a\n        Python path to a subclass of ``gunicorn.workers.base.Worker``.\n        This alternative syntax will load the gevent class:\n        ``gunicorn.workers.ggevent.GeventWorker``.\n        '