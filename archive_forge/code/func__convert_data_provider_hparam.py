import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _convert_data_provider_hparam(self, dp_hparam):
    """Builds an HParamInfo message from data provider Hyperparameter.

        Args:
          dp_hparam: The provider.Hyperparameter returned by the call to
            provider.DataProvider.list_hyperparameters().

        Returns:
          An HParamInfo to include in the Experiment.
        """
    hparam_info = api_pb2.HParamInfo(name=dp_hparam.hyperparameter_name, display_name=dp_hparam.hyperparameter_display_name, differs=dp_hparam.differs)
    if dp_hparam.domain_type == provider.HyperparameterDomainType.INTERVAL:
        hparam_info.type = api_pb2.DATA_TYPE_FLOAT64
        dp_hparam_min, dp_hparam_max = dp_hparam.domain
        hparam_info.domain_interval.min_value = dp_hparam_min
        hparam_info.domain_interval.max_value = dp_hparam_max
    elif dp_hparam.domain_type in _DISCRETE_DOMAIN_TYPE_TO_DATA_TYPE.keys():
        hparam_info.type = _DISCRETE_DOMAIN_TYPE_TO_DATA_TYPE.get(dp_hparam.domain_type)
        hparam_info.domain_discrete.extend(dp_hparam.domain)
    return hparam_info