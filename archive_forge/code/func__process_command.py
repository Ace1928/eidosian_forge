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
def _process_command(self, command):
    logger.info('Agent received command: %s' % (command['type'] if 'type' in command else 'Unknown'))
    response = {'id': command.get('id'), 'result': None}
    try:
        command_type = command['type']
        if command_type == 'run':
            result = self._command_run(command)
        elif command_type == 'stop':
            result = self._command_stop(command)
        elif command_type == 'exit':
            result = self._command_exit(command)
        elif command_type == 'resume':
            result = self._command_run(command)
        else:
            raise AgentError('No such command: %s' % command_type)
        response['result'] = result
    except Exception:
        logger.exception('Exception while processing command: %s', command)
        ex_type, ex, tb = sys.exc_info()
        response['exception'] = f'{ex_type.__name__}: {str(ex)}'
        response['traceback'] = traceback.format_tb(tb)
        del tb
    self._log.append((command, response))
    return response