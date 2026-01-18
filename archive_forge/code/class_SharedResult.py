import subprocess
import sys
from collections import namedtuple
from io import StringIO
from subprocess import PIPE
from typing import Any, Callable, Dict, Generator, Optional, Tuple
import pytest
from sphinx.testing import util
from sphinx.testing.util import SphinxTestApp, SphinxTestAppWrapperForSkipBuilding
class SharedResult:
    cache: Dict[str, Dict[str, str]] = {}

    def store(self, key: str, app_: SphinxTestApp) -> Any:
        if key in self.cache:
            return
        data = {'status': app_._status.getvalue(), 'warning': app_._warning.getvalue()}
        self.cache[key] = data

    def restore(self, key: str) -> Dict[str, StringIO]:
        if key not in self.cache:
            return {}
        data = self.cache[key]
        return {'status': StringIO(data['status']), 'warning': StringIO(data['warning'])}