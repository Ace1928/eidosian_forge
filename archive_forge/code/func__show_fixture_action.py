from typing import Generator
from typing import Optional
from typing import Union
from _pytest._io.saferepr import saferepr
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureDef
from _pytest.fixtures import SubRequest
from _pytest.scope import Scope
import pytest
def _show_fixture_action(fixturedef: FixtureDef[object], config: Config, msg: str) -> None:
    capman = config.pluginmanager.getplugin('capturemanager')
    if capman:
        capman.suspend_global_capture()
    tw = config.get_terminal_writer()
    tw.line()
    scope_indent = list(reversed(Scope)).index(fixturedef._scope)
    tw.write(' ' * 2 * scope_indent)
    tw.write('{step} {scope} {fixture}'.format(step=msg.ljust(8), scope=fixturedef.scope[0].upper(), fixture=fixturedef.argname))
    if msg == 'SETUP':
        deps = sorted((arg for arg in fixturedef.argnames if arg != 'request'))
        if deps:
            tw.write(' (fixtures used: {})'.format(', '.join(deps)))
    if hasattr(fixturedef, 'cached_param'):
        tw.write(f'[{saferepr(fixturedef.cached_param, maxsize=42)}]')
    tw.flush()
    if capman:
        capman.resume_global_capture()