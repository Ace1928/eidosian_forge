import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _compute_metric_names(self, ctx, experiment_id, hparams_run_to_tag_to_content):
    """Computes the list of metric names from all the scalar (run, tag)
        pairs.

        The return value is a list of (tag, group) pairs representing the metric
        names. The list is sorted in Python tuple-order (lexicographical).

        For example, if the scalar (run, tag) pairs are:
        ("exp/session1", "loss")
        ("exp/session2", "loss")
        ("exp/session2/eval", "loss")
        ("exp/session2/validation", "accuracy")
        ("exp/no-session", "loss_2"),
        and the runs corresponding to sessions are "exp/session1", "exp/session2",
        this method will return [("loss", ""), ("loss", "/eval"), ("accuracy",
        "/validation")]

        More precisely, each scalar (run, tag) pair is converted to a (tag, group)
        metric name, where group is the suffix of run formed by removing the
        longest prefix which is a session run. If no session run is a prefix of
        'run', the pair is skipped.

        Returns:
          A python list containing pairs. Each pair is a (tag, group) pair
          representing a metric name used in some session.
        """
    session_runs = set((run for run, tags in hparams_run_to_tag_to_content.items() if metadata.SESSION_START_INFO_TAG in tags))
    metric_names_set = set()
    scalars_run_to_tag_to_content = self.scalars_metadata(ctx, experiment_id)
    for run, tags in scalars_run_to_tag_to_content.items():
        session = _find_longest_parent_path(session_runs, run)
        if not session:
            continue
        group = os.path.relpath(run, session)
        if group == '.':
            group = ''
        metric_names_set.update(((tag, group) for tag in tags))
    metric_names_list = list(metric_names_set)
    metric_names_list.sort()
    return metric_names_list