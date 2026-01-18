import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
def _exactly_one_of(infos: Iterable[Optional['InfoType']]) -> 'InfoType':
    infos = [info for info in infos if info is not None]
    if not infos:
        raise DirectUrlValidationError('missing one of archive_info, dir_info, vcs_info')
    if len(infos) > 1:
        raise DirectUrlValidationError('more than one of archive_info, dir_info, vcs_info')
    assert infos[0] is not None
    return infos[0]