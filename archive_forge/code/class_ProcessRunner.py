import sys
import os
import time
import uuid
import shlex
import threading
import shutil
import subprocess
import logging
import inspect
import ctypes
import runpy
import requests
import psutil
import multiprocess
from dash.testing.errors import (
from dash.testing import wait
class ProcessRunner(BaseDashRunner):
    """Runs a dash application in a waitress-serve subprocess.

    This flavor is closer to production environment but slower.
    """

    def __init__(self, keep_open=False, stop_timeout=3):
        super().__init__(keep_open=keep_open, stop_timeout=stop_timeout)
        self.proc = None

    def start(self, app_module=None, application_name='app', raw_command=None, port=8050, start_timeout=3):
        """Start the server with waitress-serve in process flavor."""
        if not (app_module or raw_command):
            logging.error('the process runner needs to start with at least one valid command')
            return
        self.port = port
        args = shlex.split(raw_command if raw_command else f'waitress-serve --listen=0.0.0.0:{port} {app_module}:{application_name}.server', posix=not self.is_windows)
        logger.debug('start dash process with %s', args)
        try:
            self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wait.until(lambda: self.accessible(self.url), timeout=start_timeout)
        except (OSError, ValueError):
            logger.exception('process server has encountered an error')
            self.started = False
            self.stop()
            return
        self.started = True

    def stop(self):
        if self.proc:
            try:
                logger.info('proc.terminate with pid %s', self.proc.pid)
                self.proc.terminate()
                if self.tmp_app_path and os.path.exists(self.tmp_app_path):
                    logger.debug('removing temporary app path %s', self.tmp_app_path)
                    shutil.rmtree(self.tmp_app_path)
                _except = subprocess.TimeoutExpired
                self.proc.communicate(timeout=self.stop_timeout)
            except _except:
                logger.exception('subprocess terminate not success, trying to kill the subprocess in a safe manner')
                self.proc.kill()
                self.proc.communicate()
        logger.info('process stop completes!')