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
def _datasets_repr_(self) -> str:
    from ipywidgets import HTML, Layout, VBox
    content = []
    if self.datasets:
        for name, config in self.datasets.items():
            tab = config._tab_repr_()
            if tab:
                content.append(HTML(Template('title_data.html.j2').render(title=f'Dataset - <code>{name}</code>', data=None)))
                content.append(config._tab_repr_())
    return VBox(content, layout=Layout(width='100%'))