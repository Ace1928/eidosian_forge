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
def _tune_legacy_checkpoint_score_attr(self) -> Optional[str]:
    """Same as ``checkpoint_score_attr`` in ``tune.run``.

        Only used for Legacy API compatibility.
        """
    if self.checkpoint_score_attribute is None:
        return self.checkpoint_score_attribute
    prefix = ''
    if self.checkpoint_score_order == MIN:
        prefix = 'min-'
    return f'{prefix}{self.checkpoint_score_attribute}'