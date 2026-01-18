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
class ThreadedRunner(BaseDashRunner):
    """Runs a dash application in a thread.

    This is the default flavor to use in dash integration tests.
    """

    def __init__(self, keep_open=False, stop_timeout=3):
        super().__init__(keep_open=keep_open, stop_timeout=stop_timeout)
        self.thread = None

    def running_and_accessible(self, url):
        if self.thread.is_alive():
            return self.accessible(url)
        raise DashAppLoadingError('Thread is not alive.')

    def start(self, app, start_timeout=3, **kwargs):
        """Start the app server in threading flavor."""

        def run():
            app.scripts.config.serve_locally = True
            app.css.config.serve_locally = True
            options = kwargs.copy()
            if 'port' not in kwargs:
                options['port'] = self.port = BaseDashRunner._next_port
                BaseDashRunner._next_port += 1
            else:
                self.port = options['port']
            try:
                app.run(threaded=True, **options)
            except SystemExit:
                logger.info('Server stopped')
            except Exception as error:
                logger.exception(error)
                raise error
        retries = 0
        while not self.started and retries < 3:
            try:
                if self.thread:
                    if self.thread.is_alive():
                        self.stop()
                    else:
                        self.thread.kill()
                self.thread = KillerThread(target=run)
                self.thread.daemon = True
                self.thread.start()
                wait.until(lambda: self.running_and_accessible(self.url), timeout=start_timeout)
                self.started = self.thread.is_alive()
            except Exception as err:
                logger.exception(err)
                self.started = False
                retries += 1
                time.sleep(1)
        self.started = self.thread.is_alive()
        if not self.started:
            raise DashAppLoadingError('threaded server failed to start')

    def stop(self):
        self.thread.kill()
        self.thread.join()
        wait.until_not(self.thread.is_alive, self.stop_timeout)
        self.started = False