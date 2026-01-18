import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
def _add_task_flags_to_agent_opt(agent, opt: Opt, flags):
    """
    Handle task flags provided by the task name itself.

    With this you can set specific opts with `-t task:flag=foo`.
    """
    fl = flags.split(':')
    task = []
    for f in fl:
        if '=' in f:
            one_flag = f.split('=')
            key = one_flag[0].replace('-', '_')
            raw_value = one_flag[1].replace(';', ':')
            if raw_value.lower() == 'true':
                value = True
            elif raw_value.lower() == 'false':
                value = False
            else:
                try:
                    value = int(raw_value)
                except ValueError:
                    try:
                        value = float(raw_value)
                    except ValueError:
                        value = raw_value
            opt[key] = value
        else:
            task.append(f)
    opt['task'] = ':'.join(task)