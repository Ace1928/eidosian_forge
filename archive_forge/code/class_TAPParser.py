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
class TAPParser:

    class Plan(T.NamedTuple):
        num_tests: int
        late: bool
        skipped: bool
        explanation: T.Optional[str]

    class Bailout(T.NamedTuple):
        message: str

    class Test(T.NamedTuple):
        number: int
        name: str
        result: TestResult
        explanation: T.Optional[str]

        def __str__(self) -> str:
            return f'{self.number} {self.name}'.strip()

    class Error(T.NamedTuple):
        message: str

    class UnknownLine(T.NamedTuple):
        message: str
        lineno: int

    class Version(T.NamedTuple):
        version: int
    _MAIN = 1
    _AFTER_TEST = 2
    _YAML = 3
    _RE_BAILOUT = re.compile('Bail out!\\s*(.*)')
    _RE_DIRECTIVE = re.compile('(?:\\s*\\#\\s*([Ss][Kk][Ii][Pp]\\S*|[Tt][Oo][Dd][Oo])\\b\\s*(.*))?')
    _RE_PLAN = re.compile('1\\.\\.([0-9]+)' + _RE_DIRECTIVE.pattern)
    _RE_TEST = re.compile('((?:not )?ok)\\s*(?:([0-9]+)\\s*)?([^#]*)' + _RE_DIRECTIVE.pattern)
    _RE_VERSION = re.compile('TAP version ([0-9]+)')
    _RE_YAML_START = re.compile('(\\s+)---.*')
    _RE_YAML_END = re.compile('\\s+\\.\\.\\.\\s*')
    found_late_test = False
    bailed_out = False
    plan: T.Optional[Plan] = None
    lineno = 0
    num_tests = 0
    yaml_lineno: T.Optional[int] = None
    yaml_indent = ''
    state = _MAIN
    version = 12

    def parse_test(self, ok: bool, num: int, name: str, directive: T.Optional[str], explanation: T.Optional[str]) -> T.Generator[T.Union['TAPParser.Test', 'TAPParser.Error'], None, None]:
        name = name.strip()
        explanation = explanation.strip() if explanation else None
        if directive is not None:
            directive = directive.upper()
            if directive.startswith('SKIP'):
                if ok:
                    yield self.Test(num, name, TestResult.SKIP, explanation)
                    return
            elif directive == 'TODO':
                yield self.Test(num, name, TestResult.UNEXPECTEDPASS if ok else TestResult.EXPECTEDFAIL, explanation)
                return
            else:
                yield self.Error(f'invalid directive "{directive}"')
        yield self.Test(num, name, TestResult.OK if ok else TestResult.FAIL, explanation)

    async def parse_async(self, lines: T.AsyncIterator[str]) -> T.AsyncIterator[TYPE_TAPResult]:
        async for line in lines:
            for event in self.parse_line(line):
                yield event
        for event in self.parse_line(None):
            yield event

    def parse(self, io: T.Iterator[str]) -> T.Iterator[TYPE_TAPResult]:
        for line in io:
            yield from self.parse_line(line)
        yield from self.parse_line(None)

    def parse_line(self, line: T.Optional[str]) -> T.Iterator[TYPE_TAPResult]:
        if line is not None:
            self.lineno += 1
            line = line.rstrip()
            if self.state == self._AFTER_TEST:
                if self.version >= 13:
                    m = self._RE_YAML_START.match(line)
                    if m:
                        self.state = self._YAML
                        self.yaml_lineno = self.lineno
                        self.yaml_indent = m.group(1)
                        return
                self.state = self._MAIN
            elif self.state == self._YAML:
                if self._RE_YAML_END.match(line):
                    self.state = self._MAIN
                    return
                if line.startswith(self.yaml_indent):
                    return
                yield self.Error(f'YAML block not terminated (started on line {self.yaml_lineno})')
                self.state = self._MAIN
            assert self.state == self._MAIN
            if not line or line.startswith('#'):
                return
            m = self._RE_TEST.match(line)
            if m:
                if self.plan and self.plan.late and (not self.found_late_test):
                    yield self.Error('unexpected test after late plan')
                    self.found_late_test = True
                self.num_tests += 1
                num = self.num_tests if m.group(2) is None else int(m.group(2))
                if num != self.num_tests:
                    yield self.Error('out of order test numbers')
                yield from self.parse_test(m.group(1) == 'ok', num, m.group(3), m.group(4), m.group(5))
                self.state = self._AFTER_TEST
                return
            m = self._RE_PLAN.match(line)
            if m:
                if self.plan:
                    yield self.Error('more than one plan found')
                else:
                    num_tests = int(m.group(1))
                    skipped = num_tests == 0
                    if m.group(2):
                        if m.group(2).upper().startswith('SKIP'):
                            if num_tests > 0:
                                yield self.Error('invalid SKIP directive for plan')
                            skipped = True
                        else:
                            yield self.Error('invalid directive for plan')
                    self.plan = self.Plan(num_tests=num_tests, late=self.num_tests > 0, skipped=skipped, explanation=m.group(3))
                    yield self.plan
                return
            m = self._RE_BAILOUT.match(line)
            if m:
                yield self.Bailout(m.group(1))
                self.bailed_out = True
                return
            m = self._RE_VERSION.match(line)
            if m:
                if self.lineno != 1:
                    yield self.Error('version number must be on the first line')
                    return
                self.version = int(m.group(1))
                if self.version < 13:
                    yield self.Error('version number should be at least 13')
                else:
                    yield self.Version(version=self.version)
                return
            yield self.UnknownLine(line, self.lineno)
        else:
            if self.state == self._YAML:
                yield self.Error(f'YAML block not terminated (started on line {self.yaml_lineno})')
            if not self.bailed_out and self.plan and (self.num_tests != self.plan.num_tests):
                if self.num_tests < self.plan.num_tests:
                    yield self.Error(f'Too few tests run (expected {self.plan.num_tests}, got {self.num_tests})')
                else:
                    yield self.Error(f'Too many tests run (expected {self.plan.num_tests}, got {self.num_tests})')