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
class SingleTestRunner:

    def __init__(self, test: TestSerialisation, env: T.Dict[str, str], name: str, options: argparse.Namespace):
        self.test = test
        self.options = options
        self.cmd = self._get_cmd()
        if self.cmd and self.test.extra_paths:
            env['PATH'] = os.pathsep.join(self.test.extra_paths + ['']) + env['PATH']
            winecmd = []
            for c in self.cmd:
                winecmd.append(c)
                if os.path.basename(c).startswith('wine'):
                    env['WINEPATH'] = get_wine_shortpath(winecmd, ['Z:' + p for p in self.test.extra_paths] + env.get('WINEPATH', '').split(';'), self.test.workdir)
                    break
        if ('MALLOC_PERTURB_' not in env or not env['MALLOC_PERTURB_']) and (not options.benchmark):
            env['MALLOC_PERTURB_'] = str(random.randint(1, 255))
        if 'ASAN_OPTIONS' not in env or not env['ASAN_OPTIONS']:
            env['ASAN_OPTIONS'] = 'halt_on_error=1:abort_on_error=1:print_summary=1'
        if 'UBSAN_OPTIONS' not in env or not env['UBSAN_OPTIONS']:
            env['UBSAN_OPTIONS'] = 'halt_on_error=1:abort_on_error=1:print_summary=1:print_stacktrace=1'
        if 'MSAN_OPTIONS' not in env or not env['MSAN_OPTIONS']:
            env['UBSAN_OPTIONS'] = 'halt_on_error=1:abort_on_error=1:print_summary=1:print_stacktrace=1'
        if self.options.gdb or self.test.timeout is None or self.test.timeout <= 0:
            timeout = None
        elif self.options.timeout_multiplier is None:
            timeout = self.test.timeout
        elif self.options.timeout_multiplier <= 0:
            timeout = None
        else:
            timeout = self.test.timeout * self.options.timeout_multiplier
        is_parallel = test.is_parallel and self.options.num_processes > 1 and (not self.options.gdb)
        verbose = (test.verbose or self.options.verbose) and (not self.options.quiet)
        self.runobj = TestRun(test, env, name, timeout, is_parallel, verbose)
        if self.options.gdb:
            self.console_mode = ConsoleUser.GDB
        elif self.runobj.direct_stdout:
            self.console_mode = ConsoleUser.STDOUT
        else:
            self.console_mode = ConsoleUser.LOGGER

    def _get_test_cmd(self) -> T.Optional[T.List[str]]:
        testentry = self.test.fname[0]
        if self.options.no_rebuild and self.test.cmd_is_built and (not os.path.isfile(testentry)):
            raise TestException(f'The test program {testentry!r} does not exist. Cannot run tests before building them.')
        if testentry.endswith('.jar'):
            return ['java', '-jar'] + self.test.fname
        elif not self.test.is_cross_built and run_with_mono(testentry):
            return ['mono'] + self.test.fname
        elif self.test.cmd_is_exe and self.test.is_cross_built and self.test.needs_exe_wrapper:
            if self.test.exe_wrapper is None:
                return None
            elif self.test.cmd_is_exe:
                if not self.test.exe_wrapper.found():
                    msg = 'The exe_wrapper defined in the cross file {!r} was not found. Please check the command and/or add it to PATH.'
                    raise TestException(msg.format(self.test.exe_wrapper.name))
                return self.test.exe_wrapper.get_command() + self.test.fname
        elif self.test.cmd_is_built and (not self.test.cmd_is_exe) and is_windows():
            test_cmd = ExternalProgram._shebang_to_cmd(self.test.fname[0])
            if test_cmd is not None:
                test_cmd += self.test.fname[1:]
            return test_cmd
        return self.test.fname

    def _get_cmd(self) -> T.Optional[T.List[str]]:
        test_cmd = self._get_test_cmd()
        if not test_cmd:
            return None
        return TestHarness.get_wrapper(self.options) + test_cmd

    @property
    def is_parallel(self) -> bool:
        return self.runobj.is_parallel

    @property
    def visible_name(self) -> str:
        return self.runobj.name

    @property
    def timeout(self) -> T.Optional[int]:
        return self.runobj.timeout

    async def run(self, harness: 'TestHarness') -> TestRun:
        if self.cmd is None:
            self.stdo = 'Not run because cannot execute cross compiled binaries.'
            harness.log_start_test(self.runobj)
            self.runobj.complete_skip()
        else:
            cmd = self.cmd + self.test.cmd_args + self.options.test_args
            self.runobj.start(cmd)
            harness.log_start_test(self.runobj)
            await self._run_cmd(harness, cmd)
        return self.runobj

    async def _run_subprocess(self, args: T.List[str], *, stdout: T.Optional[int], stderr: T.Optional[int], env: T.Dict[str, str], cwd: T.Optional[str]) -> TestSubprocess:
        if self.options.gdb:
            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        def preexec_fn() -> None:
            if self.options.gdb:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            else:
                os.setsid()

        def postwait_fn() -> None:
            if self.options.gdb:
                signal.signal(signal.SIGINT, previous_sigint_handler)
        p = await asyncio.create_subprocess_exec(*args, stdout=stdout, stderr=stderr, env=env, cwd=cwd, preexec_fn=preexec_fn if not is_windows() else None)
        return TestSubprocess(p, stdout=stdout, stderr=stderr, postwait_fn=postwait_fn if not is_windows() else None)

    async def _run_cmd(self, harness: 'TestHarness', cmd: T.List[str]) -> None:
        if self.console_mode is ConsoleUser.GDB:
            stdout = None
            stderr = None
        else:
            stdout = asyncio.subprocess.PIPE
            stderr = asyncio.subprocess.STDOUT if not self.options.split and (not self.runobj.needs_parsing) else asyncio.subprocess.PIPE
        extra_cmd: T.List[str] = []
        if self.test.protocol is TestProtocol.GTEST:
            gtestname = self.test.name
            if self.test.workdir:
                gtestname = os.path.join(self.test.workdir, self.test.name)
            extra_cmd.append(f'--gtest_output=xml:{gtestname}.xml')
        p = await self._run_subprocess(cmd + extra_cmd, stdout=stdout, stderr=stderr, env=self.runobj.env, cwd=self.test.workdir)
        if self.runobj.needs_parsing:
            parse_coro = self.runobj.parse(harness, p.stdout_lines())
            parse_task = asyncio.ensure_future(parse_coro)
        else:
            parse_task = None
        stdo_task, stde_task = p.communicate(self.runobj, self.console_mode)
        await p.wait(self.runobj)
        if parse_task:
            await parse_task
        if stdo_task:
            await stdo_task
        if stde_task:
            await stde_task
        self.runobj.complete()