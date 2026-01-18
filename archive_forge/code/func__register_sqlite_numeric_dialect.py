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
@pre
def _register_sqlite_numeric_dialect(opt, file_config):
    from sqlalchemy.dialects import registry
    registry.register('sqlite.pysqlite_numeric', 'sqlalchemy.dialects.sqlite.pysqlite', '_SQLiteDialect_pysqlite_numeric')
    registry.register('sqlite.pysqlite_dollar', 'sqlalchemy.dialects.sqlite.pysqlite', '_SQLiteDialect_pysqlite_dollar')