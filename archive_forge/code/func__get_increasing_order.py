import dataclasses
import json
import os
from pathlib import Path
from typing import Dict
from typing import final
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Union
from .pathlib import resolve_from_str
from .pathlib import rm_rf
from .reports import CollectReport
from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.nodes import Directory
from _pytest.nodes import File
from _pytest.reports import TestReport
def _get_increasing_order(self, items: Iterable[nodes.Item]) -> List[nodes.Item]:
    return sorted(items, key=lambda item: item.path.stat().st_mtime, reverse=True)