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
@retry
def _TO_MOVE_create_connection(self, instance: MinecraftInstance) -> None:
    try:
        logger.debug('Creating socket connection {instance}'.format(instance=instance))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(SOCKTIME)
        sock.connect((instance.host, instance.port))
        logger.debug('Saying hello for client: {instance}'.format(instance=instance))
        self._TO_MOVE_hello(sock)
        instance.client_socket = sock
    except (socket.timeout, socket.error, ConnectionRefusedError) as e:
        instance.had_to_clean = True
        logger.error('Failed to reset (socket error), trying again!')
        logger.error('Cleaning connection! Something must have gone wrong.')
        self._TO_MOVE_clean_connection(instance)
        self._TO_MOVE_handle_frozen_minecraft(instance)
        raise e