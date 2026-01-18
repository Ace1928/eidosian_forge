import ctypes
import logging
import os
import queue
import socket
import threading
import time
import traceback
import wandb
from wandb import wandb_sdk
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps import utils as sweep_utils
class RunStatus:
    QUEUED = 'QUEUED'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    ERRORED = 'ERRORED'
    DONE = 'DONE'