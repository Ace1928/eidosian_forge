import json
import logging
import multiprocessing
import os
from collections import OrderedDict
from queue import Queue, PriorityQueue
from typing import List, Tuple, Any
import cv2
import numpy as np
from multiprocess.pool import Pool
from minerl.herobraine.hero.agent_handler import HandlerCollection, AgentHandler
from minerl.herobraine.hero.handlers import RewardHandler
@staticmethod
def _get_all_valid_recordings(path):
    directoryList = []
    if os.path.isfile(path):
        return []
    if len([f for f in os.listdir(path) if f.endswith('.mp4')]) > 0:
        if len([f for f in os.listdir(path) if f.endswith('.json')]) > 0:
            directoryList.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directoryList += DataPipelineWithReward._get_all_valid_recordings(new_path)
    directoryList = np.array(directoryList)
    np.random.shuffle(directoryList)
    return directoryList.tolist()