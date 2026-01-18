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
def _handle_event(self, event: 'ProtoEvent', history: Optional['TBHistory']=None) -> None:
    wandb.tensorboard._log(event.event, step=event.event.step, namespace=event.namespace, history=history)