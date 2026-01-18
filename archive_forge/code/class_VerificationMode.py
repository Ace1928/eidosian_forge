import enum
import os
from typing import Optional
from huggingface_hub.utils import insecure_hashlib
from .. import config
from .logging import get_logger
class VerificationMode(enum.Enum):
    """`Enum` that specifies which verification checks to run.

    The default mode is `BASIC_CHECKS`, which will perform only rudimentary checks to avoid slowdowns
    when generating/downloading a dataset for the first time.

    The verification modes:

    |                           | Verification checks                                                           |
    |---------------------------|------------------------------------------------------------------------------ |
    | `ALL_CHECKS`              | Split checks, uniqueness of the keys yielded in case of the GeneratorBuilder  |
    |                           | and the validity (number of files, checksums, etc.) of downloaded files       |
    | `BASIC_CHECKS` (default)  | Same as `ALL_CHECKS` but without checking downloaded files                    |
    | `NO_CHECKS`               | None                                                                          |

    """
    ALL_CHECKS = 'all_checks'
    BASIC_CHECKS = 'basic_checks'
    NO_CHECKS = 'no_checks'