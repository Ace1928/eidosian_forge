import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _sort_and_reduce_to_hparams_limit(experiment, hparams_limit=None):
    """Sorts and applies limit to the hparams in the given experiment proto.

    Args:
        experiment: An api_pb2.Experiment proto, which will be modified in place.
        hparams_limit: Optional number of hyperparameter metadata to include in the
            result. If unset or zero, no limit will be applied.

    Returns:
        None. `experiment` proto will be modified in place.
    """
    if not hparams_limit:
        hparams_limit = len(experiment.hparam_infos)
    limited_hparam_infos = sorted(experiment.hparam_infos, key=lambda hparam_info: (not hparam_info.differs, hparam_info.name))[:hparams_limit]
    experiment.ClearField('hparam_infos')
    experiment.hparam_infos.extend(limited_hparam_infos)