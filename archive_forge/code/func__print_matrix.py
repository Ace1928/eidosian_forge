import itertools
from pathlib import Path
from typing import Any, Dict, Optional
import typer
from thinc.api import (
from wasabi import msg
from spacy.training import Example
from spacy.util import resolve_dot_names
from .. import util
from ..schemas import ConfigSchemaTraining
from ..util import registry
from ._util import (
def _print_matrix(value):
    if value is None or isinstance(value, bool):
        return value
    result = str(value.shape) + ' - sample: '
    sample_matrix = value
    for d in range(value.ndim - 1):
        sample_matrix = sample_matrix[0]
    sample_matrix = sample_matrix[0:5]
    result = result + str(sample_matrix)
    return result