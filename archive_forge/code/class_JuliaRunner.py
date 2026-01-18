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
class JuliaRunner(ProcessRunner):

    def __init__(self, keep_open=False, stop_timeout=3):
        super().__init__(keep_open=keep_open, stop_timeout=stop_timeout)
        self.proc = None

    def start(self, app, start_timeout=30, cwd=None):
        """Start the server with subprocess and julia."""
        if os.path.isfile(app) and os.path.exists(app):
            if not cwd:
                cwd = os.path.dirname(app)
                logger.info('JuliaRunner inferred cwd from app path: %s', cwd)
        else:
            self._tmp_app_path = os.path.join('/tmp' if not self.is_windows else os.getenv('TEMP'), uuid.uuid4().hex)
            try:
                os.mkdir(self.tmp_app_path)
            except OSError:
                logger.exception('cannot make temporary folder %s', self.tmp_app_path)
            path = os.path.join(self.tmp_app_path, 'app.jl')
            logger.info('JuliaRunner start => app is Julia code chunk')
            logger.info('make a temporary Julia file for execution => %s', path)
            logger.debug('content of the Dash.jl app')
            logger.debug('%s', app)
            with open(path, 'w', encoding='utf-8') as fp:
                fp.write(app)
            app = path
            if not cwd:
                for entry in inspect.stack():
                    if '/dash/testing/' not in entry[1].replace('\\', '/'):
                        cwd = os.path.dirname(os.path.realpath(entry[1]))
                        logger.warning('get cwd from inspect => %s', cwd)
                        break
            if cwd:
                logger.info('JuliaRunner inferred cwd from the Python call stack: %s', cwd)
                assets = [os.path.join(cwd, _) for _ in os.listdir(cwd) if not _.startswith('__') and os.path.isdir(os.path.join(cwd, _))]
                for asset in assets:
                    target = os.path.join(self.tmp_app_path, os.path.basename(asset))
                    if os.path.exists(target):
                        logger.debug('delete existing target %s', target)
                        shutil.rmtree(target)
                    logger.debug('copying %s => %s', asset, self.tmp_app_path)
                    shutil.copytree(asset, target)
                    logger.debug('copied with %s', os.listdir(target))
            else:
                logger.warning('JuliaRunner found no cwd in the Python call stack. You may wish to specify an explicit working directory using something like: dashjl.run_server(app, cwd=os.path.dirname(__file__))')
        logger.info('Run Dash.jl app with julia => %s', app)
        args = shlex.split(f'julia --project {os.path.realpath(app)}', posix=not self.is_windows)
        logger.debug('start Dash.jl process with %s', args)
        try:
            self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.tmp_app_path if self.tmp_app_path else cwd)
            wait.until(lambda: self.accessible(self.url), timeout=start_timeout)
        except (OSError, ValueError):
            logger.exception('process server has encountered an error')
            self.started = False
            return
        self.started = True