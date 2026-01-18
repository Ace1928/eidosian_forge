from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import typer
from thinc.api import Config
from thinc.config import VARIABLE_RE
from wasabi import msg, table
from .. import util
from ..schemas import ConfigSchemaInit, ConfigSchemaTraining
from ..util import registry
from ._util import (
Debug a config file and show validation errors. The command will
    create all objects in the tree and validate them. Note that some config
    validation errors are blocking and will prevent the rest of the config from
    being resolved. This means that you may not see all validation errors at
    once and some issues are only shown once previous errors have been fixed.
    Similar as with the 'train' command, you can override settings from the config
    as command line options. For instance, --training.batch_size 128 overrides
    the value of "batch_size" in the block "[training]".

    DOCS: https://spacy.io/api/cli#debug-config
    