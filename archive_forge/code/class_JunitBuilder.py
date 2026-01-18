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
class JunitBuilder(TestLogger):
    """Builder for Junit test results.

    Junit is impossible to stream out, it requires attributes counting the
    total number of tests, failures, skips, and errors in the root element
    and in each test suite. As such, we use a builder class to track each
    test case, and calculate all metadata before writing it out.

    For tests with multiple results (like from a TAP test), we record the
    test as a suite with the project_name.test_name. This allows us to track
    each result separately. For tests with only one result (such as exit-code
    tests) we record each one into a suite with the name project_name. The use
    of the project_name allows us to sort subproject tests separately from
    the root project.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.root = et.Element('testsuites', tests='0', errors='0', failures='0')
        self.suites: T.Dict[str, et.Element] = {}

    def log(self, harness: 'TestHarness', test: 'TestRun') -> None:
        """Log a single test case."""
        if test.junit is not None:
            for suite in test.junit.findall('.//testsuite'):
                suite.attrib['name'] = '{}.{}.{}'.format(test.project, test.name, suite.attrib['name'])
                for case in suite.findall('.//testcase[@result]'):
                    del case.attrib['result']
                for case in suite.findall('.//testcase[@timestamp]'):
                    del case.attrib['timestamp']
                for case in suite.findall('.//testcase[@file]'):
                    del case.attrib['file']
                for case in suite.findall('.//testcase[@line]'):
                    del case.attrib['line']
                self.root.append(suite)
            return
        if test.results:
            suitename = f'{test.project}.{test.name}'
            assert suitename not in self.suites or harness.options.repeat > 1, 'duplicate suite'
            suite = self.suites[suitename] = et.Element('testsuite', name=suitename, tests=str(len(test.results)), errors=str(sum((1 for r in test.results if r.result in {TestResult.INTERRUPT, TestResult.ERROR}))), failures=str(sum((1 for r in test.results if r.result in {TestResult.FAIL, TestResult.UNEXPECTEDPASS, TestResult.TIMEOUT}))), skipped=str(sum((1 for r in test.results if r.result is TestResult.SKIP))), time=str(test.duration))
            for subtest in test.results:
                testcase = et.SubElement(suite, 'testcase', name=str(subtest), classname=suitename)
                if subtest.result is TestResult.SKIP:
                    et.SubElement(testcase, 'skipped')
                elif subtest.result is TestResult.ERROR:
                    et.SubElement(testcase, 'error')
                elif subtest.result is TestResult.FAIL:
                    et.SubElement(testcase, 'failure')
                elif subtest.result is TestResult.UNEXPECTEDPASS:
                    fail = et.SubElement(testcase, 'failure')
                    fail.text = 'Test unexpected passed.'
                elif subtest.result is TestResult.INTERRUPT:
                    fail = et.SubElement(testcase, 'error')
                    fail.text = 'Test was interrupted by user.'
                elif subtest.result is TestResult.TIMEOUT:
                    fail = et.SubElement(testcase, 'error')
                    fail.text = 'Test did not finish before configured timeout.'
                if subtest.explanation:
                    et.SubElement(testcase, 'system-out').text = subtest.explanation
            if test.stdo:
                out = et.SubElement(suite, 'system-out')
                out.text = replace_unencodable_xml_chars(test.stdo.rstrip())
            if test.stde:
                err = et.SubElement(suite, 'system-err')
                err.text = replace_unencodable_xml_chars(test.stde.rstrip())
        else:
            if test.project not in self.suites:
                suite = self.suites[test.project] = et.Element('testsuite', name=test.project, tests='1', errors='0', failures='0', skipped='0', time=str(test.duration))
            else:
                suite = self.suites[test.project]
                suite.attrib['tests'] = str(int(suite.attrib['tests']) + 1)
            testcase = et.SubElement(suite, 'testcase', name=test.name, classname=test.project, time=str(test.duration))
            if test.res is TestResult.SKIP:
                et.SubElement(testcase, 'skipped')
                suite.attrib['skipped'] = str(int(suite.attrib['skipped']) + 1)
            elif test.res is TestResult.ERROR:
                et.SubElement(testcase, 'error')
                suite.attrib['errors'] = str(int(suite.attrib['errors']) + 1)
            elif test.res is TestResult.FAIL:
                et.SubElement(testcase, 'failure')
                suite.attrib['failures'] = str(int(suite.attrib['failures']) + 1)
            if test.stdo:
                out = et.SubElement(testcase, 'system-out')
                out.text = replace_unencodable_xml_chars(test.stdo.rstrip())
            if test.stde:
                err = et.SubElement(testcase, 'system-err')
                err.text = replace_unencodable_xml_chars(test.stde.rstrip())

    async def finish(self, harness: 'TestHarness') -> None:
        """Calculate total test counts and write out the xml result."""
        for suite in self.suites.values():
            self.root.append(suite)
            for attr in ['tests', 'errors', 'failures']:
                self.root.attrib[attr] = str(int(self.root.attrib[attr]) + int(suite.attrib[attr]))
        tree = et.ElementTree(self.root)
        with open(self.filename, 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)