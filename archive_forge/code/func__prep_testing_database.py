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
def _prep_testing_database(options, file_config):
    from sqlalchemy.testing import config
    if options.dropfirst:
        from sqlalchemy.testing import provision
        for cfg in config.Config.all_configs():
            provision.drop_all_schema_objects(cfg, cfg.db)