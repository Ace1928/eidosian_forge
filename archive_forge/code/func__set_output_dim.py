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
def _set_output_dim(model, nO):
    if model.has_dim('nO') is None:
        model.set_dim('nO', nO)
    if model.has_ref('output_layer'):
        if model.get_ref('output_layer').has_dim('nO') is None:
            model.get_ref('output_layer').set_dim('nO', nO)