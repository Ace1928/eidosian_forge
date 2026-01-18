import os
import sys
from typing import Generator
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.stash import StashKey
import pytest
Cancel any traceback dumping due to an interactive exception being
    raised.