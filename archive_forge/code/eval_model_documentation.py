from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import (
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger, nice_report
from parlai.utils.world_logging import WorldLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import json
import random
from parlai.utils.distributed import (

    Evaluates a model.

    :param opt: tells the evaluation function how to run
    :return: the final result of calling report()
    