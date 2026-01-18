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
def _setup_requirements(argument):
    from sqlalchemy.testing import config
    from sqlalchemy import testing
    modname, clsname = argument.split(':')
    mod = __import__(modname)
    for component in modname.split('.')[1:]:
        mod = getattr(mod, component)
    req_cls = getattr(mod, clsname)
    config.requirements = testing.requires = req_cls()
    config.bootstrapped_as_sqlalchemy = bootstrapped_as_sqlalchemy