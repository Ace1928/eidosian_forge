import os
from typing import Dict, List
import pyarrow.fs
from ray.tune.logger import LoggerCallback
from ray.tune.experiment import Trial
from ray.tune.utils import flatten_dict
def _import_comet():
    """Try importing comet_ml.

    Used to check if comet_ml is installed and, otherwise, pass an informative
    error message.
    """
    if 'COMET_DISABLE_AUTO_LOGGING' not in os.environ:
        os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
    try:
        import comet_ml
    except ImportError:
        raise RuntimeError("pip install 'comet-ml' to use CometLoggerCallback")
    return comet_ml