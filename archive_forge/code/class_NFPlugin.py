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
class NFPlugin:
    """Plugin which implements the --nf (run new-first) option."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.active = config.option.newfirst
        assert config.cache is not None
        self.cached_nodeids = set(config.cache.get('cache/nodeids', []))

    @hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items: List[nodes.Item]) -> Generator[None, None, None]:
        res = (yield)
        if self.active:
            new_items: Dict[str, nodes.Item] = {}
            other_items: Dict[str, nodes.Item] = {}
            for item in items:
                if item.nodeid not in self.cached_nodeids:
                    new_items[item.nodeid] = item
                else:
                    other_items[item.nodeid] = item
            items[:] = self._get_increasing_order(new_items.values()) + self._get_increasing_order(other_items.values())
            self.cached_nodeids.update(new_items)
        else:
            self.cached_nodeids.update((item.nodeid for item in items))
        return res

    def _get_increasing_order(self, items: Iterable[nodes.Item]) -> List[nodes.Item]:
        return sorted(items, key=lambda item: item.path.stat().st_mtime, reverse=True)

    def pytest_sessionfinish(self) -> None:
        config = self.config
        if config.getoption('cacheshow') or hasattr(config, 'workerinput'):
            return
        if config.getoption('collectonly'):
            return
        assert config.cache is not None
        config.cache.set('cache/nodeids', sorted(self.cached_nodeids))