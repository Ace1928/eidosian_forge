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
def _send_mission(self, instance: MinecraftInstance, mission_xml_etree: etree.Element, token_in: str) -> None:
    """Sends the XML to the given instance.

        Args:
            instance (MinecraftInstance): The instance to which to send the xml

        Raises:
            socket.timeout: If the mission cannot be sent.
        """
    ok = 0
    num_retries = 0
    logger.debug('Sending mission init: {instance}'.format(instance=instance))
    while ok != 1:
        mission_xml = etree.tostring(mission_xml_etree)
        token = token_in + ':' + str(self.task.agent_count) + ':' + str(True).lower()
        if self._seed is not None:
            token += ':{}'.format(self._seed)
        token = token.encode()
        comms.send_message(instance.client_socket, mission_xml)
        comms.send_message(instance.client_socket, token)
        reply = comms.recv_message(instance.client_socket)
        ok, = struct.unpack('!I', reply)
        if ok != 1:
            num_retries += 1
            if num_retries > MAX_WAIT:
                raise socket.timeout()
            logger.debug('Recieved a MALMOBUSY from {}; trying again.'.format(instance))
            time.sleep(1)