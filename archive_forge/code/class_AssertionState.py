import sys
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from _pytest.assertion import rewrite
from _pytest.assertion import truncate
from _pytest.assertion import util
from _pytest.assertion.rewrite import assertstate_key
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
class AssertionState:
    """State for the assertion plugin."""

    def __init__(self, config: Config, mode) -> None:
        self.mode = mode
        self.trace = config.trace.root.get('assertion')
        self.hook: Optional[rewrite.AssertionRewritingHook] = None