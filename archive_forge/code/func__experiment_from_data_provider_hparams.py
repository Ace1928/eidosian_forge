import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _experiment_from_data_provider_hparams(self, data_provider_hparams):
    """Returns an experiment protobuffer based on data provider hparams.

        Args:
          data_provider_hparams: The ouput from an hparams_from_data_provider()
            call, corresponding to DataProvider.list_hyperparameters().
            A Collection[provider.Hyperparameter].

        Returns:
          The experiment proto. If there are no hyperparameters in the input,
          returns None.
        """
    if not data_provider_hparams:
        return None
    hparam_infos = [self._convert_data_provider_hparam(dp_hparam) for dp_hparam in data_provider_hparams]
    return api_pb2.Experiment(hparam_infos=hparam_infos)