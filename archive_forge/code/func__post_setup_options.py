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
@post
def _post_setup_options(opt, file_config):
    from sqlalchemy.testing import config
    config.options = options
    config.file_config = file_config