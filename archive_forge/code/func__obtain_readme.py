import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Set, Union
from ..errors import FileError, InvalidConfigError
from ..warnings import SetuptoolsWarning
from . import expand as _expand
from ._apply_pyprojecttoml import _PREVIOUSLY_DEFINED, _MissingDynamic
from ._apply_pyprojecttoml import apply as _apply
def _obtain_readme(self, dist: 'Distribution') -> Optional[Dict[str, str]]:
    if 'readme' not in self.dynamic:
        return None
    dynamic_cfg = self.dynamic_cfg
    if 'readme' in dynamic_cfg:
        return {'text': self._obtain(dist, 'readme', {}), 'content-type': dynamic_cfg['readme'].get('content-type', 'text/x-rst')}
    self._ensure_previously_set(dist, 'readme')
    return None