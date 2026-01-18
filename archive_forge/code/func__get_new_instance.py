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
def _get_new_instance(self, port=None, instance_id=None):
    """
        Gets a new instance and sets up a logger if need be. 
        """
    if port is not None:
        instance = InstanceManager.add_existing_instance(port)
    else:
        instance = InstanceManager.get_instance(os.getpid(), instance_id=instance_id)
    if InstanceManager.is_remote():
        launch_queue_logger_thread(instance, self.is_closed)
    instance.launch(replaceable=self._is_fault_tolerant)
    instance.had_to_clean = False
    return instance