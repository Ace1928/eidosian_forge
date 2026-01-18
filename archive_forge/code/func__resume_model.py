import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union
import srsly
from thinc.api import (
from thinc.config import ConfigValidationError
from wasabi import Printer
from ..errors import Errors
from ..schemas import ConfigSchemaPretrain
from ..tokens import Doc
from ..util import dot_to_object, load_model_from_config, registry
from .example import Example
def _resume_model(model: Model, resume_path: Path, epoch_resume: Optional[int], silent: bool=True) -> int:
    msg = Printer(no_print=silent)
    msg.info(f'Resume training tok2vec from: {resume_path}')
    with resume_path.open('rb') as file_:
        weights_data = file_.read()
        model.get_ref('tok2vec').from_bytes(weights_data)
    if epoch_resume is None:
        model_name = re.search('model\\d+\\.bin', str(resume_path))
        if model_name:
            epoch_resume = int(model_name.group(0)[5:][:-4]) + 1
        else:
            raise ValueError(Errors.E1020)
    msg.info(f'Resuming from epoch: {epoch_resume}')
    return epoch_resume