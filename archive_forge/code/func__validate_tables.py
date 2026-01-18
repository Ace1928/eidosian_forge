import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from thinc.api import Model
from .. import util
from ..errors import Errors, Warnings
from ..language import Language
from ..lookups import Lookups, load_lookups
from ..scorer import Scorer
from ..tokens import Doc, Token
from ..training import Example
from ..util import SimpleFrozenList, logger, registry
from ..vocab import Vocab
from .pipe import Pipe
def _validate_tables(self, error_message: str=Errors.E912) -> None:
    """Check that the lookups are correct for the current mode."""
    required_tables, optional_tables = self.get_lookups_config(self.mode)
    for table in required_tables:
        if table not in self.lookups:
            raise ValueError(error_message.format(mode=self.mode, tables=required_tables, found=self.lookups.tables))
    self._validated = True