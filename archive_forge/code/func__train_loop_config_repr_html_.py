import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import ray
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.air.config import RunConfig, ScalingConfig
from ray.train import BackendConfig, Checkpoint, TrainingIterator
from ray.train._internal import session
from ray.train._internal.backend_executor import BackendExecutor, TrialInfo
from ray.train._internal.data_config import DataConfig
from ray.train._internal.session import _TrainingResult, get_session
from ray.train._internal.utils import construct_train_func
from ray.train.trainer import BaseTrainer, GenDataset
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
def _train_loop_config_repr_html_(self) -> str:
    if self._train_loop_config:
        table_data = {}
        for k, v in self._train_loop_config.items():
            if isinstance(v, str) or str(v).isnumeric():
                table_data[k] = v
            elif hasattr(v, '_repr_html_'):
                table_data[k] = v._repr_html_()
            else:
                table_data[k] = str(v)
        return Template('title_data.html.j2').render(title='Train Loop Config', data=Template('scrollableTable.html.j2').render(table=tabulate(table_data.items(), headers=['Setting', 'Value'], showindex=False, tablefmt='unsafehtml'), max_height='none'))
    else:
        return ''