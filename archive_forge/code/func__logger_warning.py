import traceback
from copy import deepcopy
import json
import logging
from minerl.env.comms import retry
from minerl.env.exceptions import MissionInitException
import os
from minerl.herobraine.wrapper import EnvWrapper
import struct
from minerl.env.malmo import InstanceManager, MinecraftInstance, launch_queue_logger_thread, malmo_version
import uuid
import coloredlogs
import gym
import socket
import time
from lxml import etree
from minerl.env import comms
import xmltodict
from concurrent.futures import ThreadPoolExecutor
import cv2
from minerl.herobraine.env_spec import EnvSpec
from typing import Any, Callable, Dict, List, Optional, Tuple
def _logger_warning(self, message, *args, once=False, **kwargs):
    if once:
        if not hasattr(self, 'silenced_logs'):
            self.silenced_logs = set()
        import hashlib
        import traceback
        stack = traceback.extract_stack()
        locator = f'{stack[-2].filename}:{stack[-2].lineno}'
        key = hashlib.md5(locator.encode('utf-8')).hexdigest()
        if key in self.silenced_logs:
            return
        self.silenced_logs.add(key)
    logger.warning(message, *args, **kwargs)