import logging
import multiprocessing
import os
import platform
import queue
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
import yaml
import wandb
from wandb import util, wandb_lib, wandb_sdk
from wandb.agents.pyagent import pyagent
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps import utils as sweep_utils
class AgentProcess:
    """Launch and manage a process."""

    def __init__(self, env=None, command=None, function=None, run_id=None, in_jupyter=None):
        self._popen = None
        self._proc = None
        self._finished_q = multiprocessing.Queue()
        self._proc_killed = False
        if command:
            if platform.system() == 'Windows':
                kwargs = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                kwargs = dict(preexec_fn=os.setpgrp)
            self._popen = subprocess.Popen(command, env=env, **kwargs)
        elif function:
            self._proc = multiprocessing.Process(target=self._start, args=(self._finished_q, env, function, run_id, in_jupyter))
            self._proc.start()
        else:
            raise AgentError('Agent Process requires command or function')

    def _start(self, finished_q, env, function, run_id, in_jupyter):
        if env:
            for k, v in env.items():
                os.environ[k] = v
        print('wandb: Agent Started Run:', run_id)
        if function:
            function()
        print('wandb: Agent Finished Run:', run_id, '\n')
        run = wandb.run
        if run:
            wandb.join()
        finished_q.put(True)

    def poll(self):
        if self._popen:
            return self._popen.poll()
        if self._proc_killed:
            self._proc.join()
            return True
        try:
            finished = self._finished_q.get(False, 0)
            if finished:
                return True
        except queue.Empty:
            pass
        return

    def wait(self):
        if self._popen:
            if platform.system() == 'Windows':
                try:
                    while True:
                        p = self._popen.poll()
                        if p is not None:
                            return p
                        time.sleep(1)
                except KeyboardInterrupt:
                    raise
            return self._popen.wait()
        return self._proc.join()

    def kill(self):
        if self._popen:
            return self._popen.kill()
        pid = self._proc.pid
        if pid:
            ret = os.kill(pid, signal.SIGKILL)
            self._proc_killed = True
            return ret
        return

    def terminate(self):
        if self._popen:
            if platform.system() == 'Windows':
                return self._popen.send_signal(signal.CTRL_C_EVENT)
            return self._popen.terminate()
        return self._proc.terminate()