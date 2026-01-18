import json
import numpy as np
import os
import signal
from typing import Dict
from parlai.core.metrics import Metric
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.exceptions import StopTrainException
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import aggregate_named_reports, aggregate_unnamed_reports
from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.worlds import create_task
from parlai.scripts.build_dict import build_dict, setup_args as setup_dict_args
from parlai.utils.distributed import (
from parlai.utils.misc import Timer, nice_report
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
def _sync_metrics(self, metrics):
    """
        Sync training metrics across workers.

        A handful of special cases are handled as exceptions, and the remaining metrics
        are simply averaged across workers.
        """
    if not is_distributed():
        return metrics
    all_versions = all_gather_list(metrics)
    return aggregate_unnamed_reports(all_versions)