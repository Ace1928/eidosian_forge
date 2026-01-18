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
def _session_groups_from_tags(self):
    """Constructs lists of SessionGroups based on hparam tag metadata."""
    hparams_run_to_tag_to_content = self._backend_context.hparams_metadata(self._request_context, self._experiment_id)
    experiment = self._backend_context.experiment_from_metadata(self._request_context, self._experiment_id, hparams_run_to_tag_to_content, [])
    extractors = _create_extractors(self._request.col_params)
    filters = _create_filters(self._request.col_params, extractors)
    session_groups = self._build_session_groups(hparams_run_to_tag_to_content, experiment)
    session_groups = self._filter(session_groups, filters)
    self._sort(session_groups, extractors)
    return session_groups