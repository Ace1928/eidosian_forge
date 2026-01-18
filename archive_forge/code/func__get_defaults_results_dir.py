import os
from pathlib import Path
from ray.air.constants import (  # noqa: F401
def _get_defaults_results_dir() -> str:
    return os.environ.get('RAY_AIR_LOCAL_CACHE_DIR') or os.environ.get('TEST_TMPDIR') or os.environ.get('TUNE_RESULT_DIR') or Path('~/ray_results').expanduser().as_posix()