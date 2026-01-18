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

    Analyze a Thinc model implementation. Includes checks for internal structure
    and activations during training.

    DOCS: https://spacy.io/api/cli#debug-model
    