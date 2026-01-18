from __future__ import annotations
import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
from sqlalchemy.testing import asyncio
def _setup_engine(cls):
    if getattr(cls, '__engine_options__', None):
        opts = dict(cls.__engine_options__)
        opts['scope'] = 'class'
        eng = engines.testing_engine(options=opts)
        config._current.push_engine(eng, testing)