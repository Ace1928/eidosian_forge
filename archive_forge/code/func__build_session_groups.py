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
def _build_session_groups(self, hparams_run_to_tag_to_content, experiment):
    """Returns a list of SessionGroups protobuffers from the summary
        data."""
    groups_by_name = {}
    session_names = [run for run, tags in hparams_run_to_tag_to_content.items() if metadata.SESSION_START_INFO_TAG in tags]
    metric_runs = set()
    metric_tags = set()
    for session_name in session_names:
        for metric in experiment.metric_infos:
            metric_name = metric.name
            run, tag = metrics.run_tag_from_session_and_metric(session_name, metric_name)
            metric_runs.add(run)
            metric_tags.add(tag)
    all_metric_evals = self._backend_context.read_last_scalars(self._request_context, self._experiment_id, run_tag_filter=provider.RunTagFilter(runs=metric_runs, tags=metric_tags))
    for session_name, tag_to_content in hparams_run_to_tag_to_content.items():
        if metadata.SESSION_START_INFO_TAG not in tag_to_content:
            continue
        start_info = metadata.parse_session_start_info_plugin_data(tag_to_content[metadata.SESSION_START_INFO_TAG])
        end_info = None
        if metadata.SESSION_END_INFO_TAG in tag_to_content:
            end_info = metadata.parse_session_end_info_plugin_data(tag_to_content[metadata.SESSION_END_INFO_TAG])
        session = self._build_session(experiment, session_name, start_info, end_info, all_metric_evals)
        if session.status in self._request.allowed_statuses:
            self._add_session(session, start_info, groups_by_name)
    groups = groups_by_name.values()
    for group in groups:
        group.sessions.sort(key=operator.attrgetter('name'))
        self._aggregate_metrics(group)
    return groups