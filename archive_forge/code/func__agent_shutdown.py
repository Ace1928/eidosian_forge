import json
import os
import random
import time
import copy
import numpy as np
import pickle
from joblib import Parallel, delayed
from parlai.core.worlds import MultiAgentDialogWorld
from parlai.mturk.core.agents import MTURK_DISCONNECT_MESSAGE
from parlai.mturk.core.worlds import MTurkOnboardWorld
def _agent_shutdown(agent, timeout):
    agent.shutdown(timeout=timeout)