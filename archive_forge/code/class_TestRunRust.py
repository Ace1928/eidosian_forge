from __future__ import annotations
from pathlib import Path
from collections import deque
from contextlib import suppress
from copy import deepcopy
from fnmatch import fnmatch
import argparse
import asyncio
import datetime
import enum
import json
import multiprocessing
import os
import pickle
import platform
import random
import re
import signal
import subprocess
import shlex
import sys
import textwrap
import time
import typing as T
import unicodedata
import xml.etree.ElementTree as et
from . import build
from . import environment
from . import mlog
from .coredata import MesonVersionMismatchException, major_versions_differ
from .coredata import version as coredata_version
from .mesonlib import (MesonException, OptionKey, OrderedSet, RealPathAction,
from .mintro import get_infodir, load_info_file
from .programs import ExternalProgram
from .backend.backends import TestProtocol, TestSerialisation
class TestRunRust(TestRun):

    @property
    def needs_parsing(self) -> bool:
        return True

    async def parse(self, harness: 'TestHarness', lines: T.AsyncIterator[str]) -> None:

        def parse_res(n: int, name: str, result: str) -> TAPParser.Test:
            if result == 'ok':
                return TAPParser.Test(n, name, TestResult.OK, None)
            elif result == 'ignored':
                return TAPParser.Test(n, name, TestResult.SKIP, None)
            elif result == 'FAILED':
                return TAPParser.Test(n, name, TestResult.FAIL, None)
            return TAPParser.Test(n, name, TestResult.ERROR, f'Unsupported output from rust test: {result}')
        n = 1
        async for line in lines:
            if line.startswith('test ') and (not line.startswith('test result')):
                _, name, _, result = line.rstrip().split(' ')
                name = name.replace('::', '.')
                t = parse_res(n, name, result)
                self.results.append(t)
                harness.log_subtest(self, name, t.result)
                n += 1
        res = None
        if all((t.result is TestResult.SKIP for t in self.results)):
            res = TestResult.SKIP
        elif any((t.result is TestResult.ERROR for t in self.results)):
            res = TestResult.ERROR
        elif any((t.result is TestResult.FAIL for t in self.results)):
            res = TestResult.FAIL
        if res and self.res == TestResult.RUNNING:
            self.res = res