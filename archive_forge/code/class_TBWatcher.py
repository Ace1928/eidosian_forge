import glob
import logging
import os
import queue
import socket
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib import filesystem
from wandb.viz import CustomChart
from . import run as internal_run
class TBWatcher:
    _logdirs: 'Dict[str, TBDirWatcher]'
    _watcher_queue: 'PriorityQueue'

    def __init__(self, settings: 'SettingsStatic', run_proto: 'RunRecord', interface: 'InterfaceQueue', force: bool=False) -> None:
        self._logdirs = {}
        self._consumer: Optional[TBEventConsumer] = None
        self._settings = settings
        self._interface = interface
        self._run_proto = run_proto
        self._force = force
        self._watcher_queue = queue.PriorityQueue()
        wandb.tensorboard.reset_state()

    def _calculate_namespace(self, logdir: str, rootdir: str) -> Optional[str]:
        namespace: Optional[str]
        dirs = list(self._logdirs) + [logdir]
        if os.path.isfile(logdir):
            filename = os.path.basename(logdir)
        else:
            filename = ''
        if rootdir == '':
            rootdir = util.to_forward_slash_path(os.path.dirname(os.path.commonprefix(dirs)))
            namespace = logdir.replace(filename, '').replace(rootdir, '').strip('/')
            if len(dirs) == 1 and namespace not in ['train', 'validation']:
                namespace = None
        else:
            namespace = logdir.replace(filename, '').replace(rootdir, '').strip('/')
        return namespace

    def add(self, logdir: str, save: bool, root_dir: str) -> None:
        logdir = util.to_forward_slash_path(logdir)
        root_dir = util.to_forward_slash_path(root_dir)
        if logdir in self._logdirs:
            return
        namespace = self._calculate_namespace(logdir, root_dir)
        if not self._consumer:
            self._consumer = TBEventConsumer(self, self._watcher_queue, self._run_proto, self._settings)
            self._consumer.start()
        tbdir_watcher = TBDirWatcher(self, logdir, save, namespace, self._watcher_queue, self._force)
        self._logdirs[logdir] = tbdir_watcher
        tbdir_watcher.start()

    def finish(self) -> None:
        for tbdirwatcher in self._logdirs.values():
            tbdirwatcher.shutdown()
        for tbdirwatcher in self._logdirs.values():
            tbdirwatcher.finish()
        if self._consumer:
            self._consumer.finish()