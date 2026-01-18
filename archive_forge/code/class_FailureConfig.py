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
@dataclass
@PublicAPI(stability='stable')
class FailureConfig:
    """Configuration related to failure handling of each training/tuning run.

    Args:
        max_failures: Tries to recover a run at least this many times.
            Will recover from the latest checkpoint if present.
            Setting to -1 will lead to infinite recovery retries.
            Setting to 0 will disable retries. Defaults to 0.
        fail_fast: Whether to fail upon the first error.
            If fail_fast='raise' provided, the original error during training will be
            immediately raised. fail_fast='raise' can easily leak resources and
            should be used with caution.
    """
    max_failures: int = 0
    fail_fast: Union[bool, str] = False

    def __post_init__(self):
        if not (isinstance(self.fail_fast, bool) or self.fail_fast.upper() == 'RAISE'):
            raise ValueError(f"fail_fast must be one of {{bool, 'raise'}}. Got {self.fail_fast}.")
        if self.fail_fast and self.max_failures != 0:
            raise ValueError(f'max_failures must be 0 if fail_fast={repr(self.fail_fast)}.')

    def __repr__(self):
        return _repr_dataclass(self)

    def _repr_html_(self):
        return Template('scrollableTable.html.j2').render(table=tabulate({'Setting': ['Max failures', 'Fail fast'], 'Value': [self.max_failures, self.fail_fast]}, tablefmt='html', showindex=False, headers='keys'), max_height='none')