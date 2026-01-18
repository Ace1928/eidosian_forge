import collections
import dataclasses
import operator
import re
from typing import Optional
from google.protobuf import struct_pb2
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import metrics
def _build_session_metric_values(self, experiment, session_name, all_metric_evals):
    """Builds the session metric values."""
    result = []
    for metric_info in experiment.metric_infos:
        metric_name = metric_info.name
        run, tag = metrics.run_tag_from_session_and_metric(session_name, metric_name)
        datum = all_metric_evals.get(run, {}).get(tag)
        if not datum:
            continue
        result.append(api_pb2.MetricValue(name=metric_name, wall_time_secs=datum.wall_time, training_step=datum.step, value=datum.value))
    return result