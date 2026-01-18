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
def _get_docs(lang: str='en'):
    nlp = util.get_lang_class(lang)()
    return list(nlp.pipe(_sentences()))