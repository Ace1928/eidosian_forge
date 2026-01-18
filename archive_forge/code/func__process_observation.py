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
def _process_observation(self, actor_name, pov, info) -> Dict[str, Any]:
    """
        Process observation into the proper dict space.
        """
    if info:
        info = json.loads(info)
    else:
        info = {}
    info['pov'] = pov
    bottom_env_spec = self.task
    while isinstance(bottom_env_spec, EnvWrapper):
        bottom_env_spec = bottom_env_spec.env_to_wrap
    obs_dict = {}
    monitor_dict = {}
    for h in bottom_env_spec.observables:
        obs_dict[h.to_string()] = h.from_hero(info)
    if isinstance(self.task, EnvWrapper):
        obs_dict = self.task.wrap_observation(obs_dict)
    self._last_pov[actor_name] = obs_dict['pov']
    self._last_obs[actor_name] = obs_dict
    for m in self.task.monitors:
        monitor_dict[m.to_string()] = m.from_hero(info)
    return (obs_dict, monitor_dict)