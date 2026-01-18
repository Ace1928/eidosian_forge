import logging
from collections import defaultdict
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import (
import pyarrow.fs
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import PublicAPI, Deprecated
from ray.widgets import Template, make_table_html_repr
from ray.data.preprocessor import Preprocessor
@property
def _resources_per_worker_not_none(self):
    if self.resources_per_worker is None:
        if self.use_gpu:
            return {'GPU': 1}
        else:
            return {'CPU': 1}
    resources_per_worker = {k: v for k, v in self.resources_per_worker.items() if v != 0}
    resources_per_worker.setdefault('GPU', int(self.use_gpu))
    return resources_per_worker