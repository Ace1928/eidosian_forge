import re
from typing import List, Optional, Tuple
from pip._vendor.packaging.tags import (
def _get_custom_interpreter(implementation: Optional[str]=None, version: Optional[str]=None) -> str:
    if implementation is None:
        implementation = interpreter_name()
    if version is None:
        version = interpreter_version()
    return f'{implementation}{version}'