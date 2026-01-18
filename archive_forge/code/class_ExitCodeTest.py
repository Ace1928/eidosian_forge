import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
class ExitCodeTest(fixtures.TestWithFixtures):

    def setUp(self) -> None:
        out = sinks.BufferFixture()
        self.useFixture(out)
        self.ap = autopage.AutoPager(out.stream)

    def test_success(self) -> None:
        with self.ap:
            pass
        self.assertEqual(0, self.ap.exit_code())

    def test_pager_broken_pipe_flush(self) -> None:
        flush = mock.MagicMock(side_effect=BrokenPipeError)
        with sinks.TTYFixture() as out:
            ap = autopage.AutoPager(out.stream)
            with fixtures.MockPatch('subprocess.Popen') as popen:
                with sinks.BufferFixture() as pager_in:
                    popen.mock.return_value.stdin = pager_in.stream
                    with ap as stream:
                        stream.write('foo')
                        stream.close = flush
            self.assertEqual(141, ap.exit_code())

    def test_no_pager_broken_pipe_flush(self) -> None:
        flush = mock.MagicMock(side_effect=BrokenPipeError)
        with self.ap as stream:
            stream.write('foo')
            stream.flush = flush
        self.assertEqual(141, self.ap.exit_code())

    def test_broken_pipe(self) -> None:
        with self.ap:
            raise BrokenPipeError
        self.assertEqual(141, self.ap.exit_code())

    def test_exception(self) -> None:

        class MyException(Exception):
            pass

        def run() -> None:
            with self.ap:
                raise MyException
        self.assertRaises(MyException, run)
        self.assertEqual(1, self.ap.exit_code())

    def test_base_exception(self) -> None:

        class MyBaseException(BaseException):
            pass

        def run() -> None:
            with self.ap:
                raise MyBaseException
        self.assertRaises(MyBaseException, run)
        self.assertEqual(1, self.ap.exit_code())

    def test_interrupt(self) -> None:

        def run() -> None:
            with self.ap:
                raise KeyboardInterrupt
        self.assertRaises(KeyboardInterrupt, run)
        self.assertEqual(130, self.ap.exit_code())

    def test_system_exit(self) -> None:

        def run() -> None:
            with self.ap:
                raise SystemExit(42)
        self.assertRaises(SystemExit, run)
        self.assertEqual(42, self.ap.exit_code())