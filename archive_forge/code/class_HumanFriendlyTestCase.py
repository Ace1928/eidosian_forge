import datetime
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import unittest
import warnings
from humanfriendly import (
from humanfriendly.case import CaseInsensitiveDict, CaseInsensitiveKey
from humanfriendly.cli import main
from humanfriendly.compat import StringIO
from humanfriendly.decorators import cached
from humanfriendly.deprecation import DeprecationProxy, define_aliases, deprecated_args, get_aliases
from humanfriendly.prompts import (
from humanfriendly.sphinx import (
from humanfriendly.tables import (
from humanfriendly.terminal import (
from humanfriendly.terminal.html import html_to_ansi
from humanfriendly.terminal.spinners import AutomaticSpinner, Spinner
from humanfriendly.testing import (
from humanfriendly.text import (
from humanfriendly.usage import (
from mock import MagicMock
class HumanFriendlyTestCase(TestCase):
    """Container for the `humanfriendly` test suite."""

    def test_case_insensitive_dict(self):
        """Test the CaseInsensitiveDict class."""
        assert len(CaseInsensitiveDict([('key', True), ('KEY', False)])) == 1
        assert len(CaseInsensitiveDict([('one', True), ('ONE', False)], one=False, two=True)) == 2
        assert len(CaseInsensitiveDict(dict(key=True, KEY=False))) == 1
        assert len(CaseInsensitiveDict(dict(one=True, ONE=False), one=False, two=True)) == 2
        assert len(CaseInsensitiveDict(one=True, ONE=False, two=True)) == 2
        obj = CaseInsensitiveDict.fromkeys(['One', 'one', 'ONE', 'Two', 'two', 'TWO'])
        assert len(obj) == 2
        obj = CaseInsensitiveDict(existing_key=42)
        assert obj.get('Existing_Key') == 42
        obj = CaseInsensitiveDict(existing_key=42)
        assert obj.pop('Existing_Key') == 42
        assert len(obj) == 0
        obj = CaseInsensitiveDict(existing_key=42)
        assert obj.setdefault('Existing_Key') == 42
        obj.setdefault('other_key', 11)
        assert obj['Other_Key'] == 11
        obj = CaseInsensitiveDict(existing_key=42)
        assert 'Existing_Key' in obj
        obj = CaseInsensitiveDict(existing_key=42)
        del obj['Existing_Key']
        assert len(obj) == 0
        obj = CaseInsensitiveDict(existing_key=42)
        assert obj['Existing_Key'] == 42
        obj = CaseInsensitiveDict(existing_key=42)
        obj['Existing_Key'] = 11
        assert obj['existing_key'] == 11

    def test_case_insensitive_key(self):
        """Test the CaseInsensitiveKey class."""
        polite = CaseInsensitiveKey("Please don't shout")
        rude = CaseInsensitiveKey("PLEASE DON'T SHOUT")
        assert polite == rude
        mapping = {}
        mapping[polite] = 1
        mapping[rude] = 2
        assert len(mapping) == 1

    def test_capture_output(self):
        """Test the CaptureOutput class."""
        with CaptureOutput() as capturer:
            sys.stdout.write('Something for stdout.\n')
            sys.stderr.write('And for stderr.\n')
            assert capturer.stdout.get_lines() == ['Something for stdout.']
            assert capturer.stderr.get_lines() == ['And for stderr.']

    def test_skip_on_raise(self):
        """Test the skip_on_raise() decorator."""

        def test_fn():
            raise NotImplementedError()
        decorator_fn = skip_on_raise(NotImplementedError)
        decorated_fn = decorator_fn(test_fn)
        self.assertRaises(NotImplementedError, test_fn)
        self.assertRaises(unittest.SkipTest, decorated_fn)

    def test_retry_raise(self):
        """Test :func:`~humanfriendly.testing.retry()` based on assertion errors."""

        def success_helper():
            if not hasattr(success_helper, 'was_called'):
                setattr(success_helper, 'was_called', True)
                assert False
            else:
                return 'yes'
        assert retry(success_helper) == 'yes'

        def failure_helper():
            assert False
        with self.assertRaises(AssertionError):
            retry(failure_helper, timeout=1)

    def test_retry_return(self):
        """Test :func:`~humanfriendly.testing.retry()` based on return values."""

        def success_helper():
            if not hasattr(success_helper, 'was_called'):
                setattr(success_helper, 'was_called', True)
                return False
            else:
                return 42
        assert retry(success_helper) == 42
        with self.assertRaises(CallableTimedOut):
            retry(lambda: False, timeout=1)

    def test_mocked_program(self):
        """Test :class:`humanfriendly.testing.MockedProgram`."""
        name = random_string()
        script = dedent('\n            # This goes to stdout.\n            tr a-z A-Z\n            # This goes to stderr.\n            echo Fake warning >&2\n        ')
        with MockedProgram(name=name, returncode=42, script=script) as directory:
            assert os.path.isdir(directory)
            assert os.path.isfile(os.path.join(directory, name))
            program = subprocess.Popen(name, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = program.communicate(input=b'hello world\n')
            assert program.returncode == 42
            assert stdout == b'HELLO WORLD\n'
            assert stderr == b'Fake warning\n'

    def test_temporary_directory(self):
        """Test :class:`humanfriendly.testing.TemporaryDirectory`."""
        with TemporaryDirectory() as directory:
            assert os.path.isdir(directory)
            temporary_file = os.path.join(directory, 'some-file')
            with open(temporary_file, 'w') as handle:
                handle.write('Hello world!')
        assert not os.path.exists(temporary_file)
        assert not os.path.exists(directory)

    def test_touch(self):
        """Test :func:`humanfriendly.testing.touch()`."""
        with TemporaryDirectory() as directory:
            filename = os.path.join(directory, random_string())
            assert not os.path.isfile(filename)
            touch(filename)
            assert os.path.isfile(filename)
            filename = os.path.join(directory, random_string(), random_string())
            assert not os.path.isfile(filename)
            touch(filename)
            assert os.path.isfile(filename)

    def test_patch_attribute(self):
        """Test :class:`humanfriendly.testing.PatchedAttribute`."""

        class Subject(object):
            my_attribute = 42
        instance = Subject()
        assert instance.my_attribute == 42
        with PatchedAttribute(instance, 'my_attribute', 13) as return_value:
            assert return_value is instance
            assert instance.my_attribute == 13
        assert instance.my_attribute == 42

    def test_patch_item(self):
        """Test :class:`humanfriendly.testing.PatchedItem`."""
        instance = dict(my_item=True)
        assert instance['my_item'] is True
        with PatchedItem(instance, 'my_item', False) as return_value:
            assert return_value is instance
            assert instance['my_item'] is False
        assert instance['my_item'] is True

    def test_run_cli_intercepts_exit(self):
        """Test that run_cli() intercepts SystemExit."""
        returncode, output = run_cli(lambda: sys.exit(42))
        self.assertEqual(returncode, 42)

    def test_run_cli_intercepts_error(self):
        """Test that run_cli() intercepts exceptions."""
        returncode, output = run_cli(self.run_cli_raise_other)
        self.assertEqual(returncode, 1)

    def run_cli_raise_other(self):
        """run_cli() sample that raises an exception."""
        raise ValueError()

    def test_run_cli_intercepts_output(self):
        """Test that run_cli() intercepts output."""
        expected_output = random_string() + '\n'
        returncode, output = run_cli(lambda: sys.stdout.write(expected_output))
        self.assertEqual(returncode, 0)
        self.assertEqual(output, expected_output)

    def test_caching_decorator(self):
        """Test the caching decorator."""
        a = cached(lambda: random.random())
        b = cached(lambda: random.random())
        assert a() == a()
        assert b() == b()
        assert a() != b()

    def test_compact(self):
        """Test :func:`humanfriendly.text.compact()`."""
        assert compact(' a \n\n b ') == 'a b'
        assert compact('\n            %s template notation\n        ', 'Simple') == 'Simple template notation'
        assert compact('\n            More {type} template notation\n        ', type='readable') == 'More readable template notation'

    def test_compact_empty_lines(self):
        """Test :func:`humanfriendly.text.compact_empty_lines()`."""
        assert compact_empty_lines('foo') == 'foo'
        assert compact_empty_lines('\tfoo') == '\tfoo'
        assert compact_empty_lines('foo\nbar') == 'foo\nbar'
        assert compact_empty_lines('foo\n\nbar') == 'foo\n\nbar'
        assert compact_empty_lines('foo\n\n\nbar') == 'foo\n\nbar'
        assert compact_empty_lines('foo\n\n\n\nbar') == 'foo\n\nbar'
        assert compact_empty_lines('foo\n\n\n\n\nbar') == 'foo\n\nbar'

    def test_dedent(self):
        """Test :func:`humanfriendly.text.dedent()`."""
        assert dedent('\n line 1\n  line 2\n\n') == 'line 1\n line 2\n'
        assert dedent('\n            Dedented, %s text\n        ', 'interpolated') == 'Dedented, interpolated text\n'
        assert dedent('\n            Dedented, {op} text\n        ', op='formatted') == 'Dedented, formatted text\n'

    def test_pluralization(self):
        """Test :func:`humanfriendly.text.pluralize()`."""
        assert pluralize(1, 'word') == '1 word'
        assert pluralize(2, 'word') == '2 words'
        assert pluralize(1, 'box', 'boxes') == '1 box'
        assert pluralize(2, 'box', 'boxes') == '2 boxes'

    def test_generate_slug(self):
        """Test :func:`humanfriendly.text.generate_slug()`."""
        self.assertEqual('some-random-text', generate_slug('Some Random Text!'))
        self.assertEqual('some-random-text', generate_slug('some-random-text'))
        with self.assertRaises(ValueError):
            generate_slug(' ')
        with self.assertRaises(ValueError):
            generate_slug('-')

    def test_boolean_coercion(self):
        """Test :func:`humanfriendly.coerce_boolean()`."""
        for value in [True, 'TRUE', 'True', 'true', 'on', 'yes', '1']:
            self.assertEqual(True, coerce_boolean(value))
        for value in [False, 'FALSE', 'False', 'false', 'off', 'no', '0']:
            self.assertEqual(False, coerce_boolean(value))
        with self.assertRaises(ValueError):
            coerce_boolean('not a boolean')

    def test_pattern_coercion(self):
        """Test :func:`humanfriendly.coerce_pattern()`."""
        empty_pattern = re.compile('')
        assert isinstance(coerce_pattern('foobar'), type(empty_pattern))
        assert empty_pattern is coerce_pattern(empty_pattern)
        pattern = coerce_pattern('foobar', re.IGNORECASE)
        assert pattern.match('FOOBAR')
        with self.assertRaises(ValueError):
            coerce_pattern([])

    def test_format_timespan(self):
        """Test :func:`humanfriendly.format_timespan()`."""
        minute = 60
        hour = minute * 60
        day = hour * 24
        week = day * 7
        year = week * 52
        assert '1 nanosecond' == format_timespan(1e-09, detailed=True)
        assert '500 nanoseconds' == format_timespan(5e-07, detailed=True)
        assert '1 microsecond' == format_timespan(1e-06, detailed=True)
        assert '500 microseconds' == format_timespan(0.0005, detailed=True)
        assert '1 millisecond' == format_timespan(0.001, detailed=True)
        assert '500 milliseconds' == format_timespan(0.5, detailed=True)
        assert '0.5 seconds' == format_timespan(0.5, detailed=False)
        assert '0 seconds' == format_timespan(0)
        assert '0.54 seconds' == format_timespan(0.54321)
        assert '1 second' == format_timespan(1)
        assert '3.14 seconds' == format_timespan(math.pi)
        assert '1 minute' == format_timespan(minute)
        assert '1 minute and 20 seconds' == format_timespan(80)
        assert '2 minutes' == format_timespan(minute * 2)
        assert '1 hour' == format_timespan(hour)
        assert '2 hours' == format_timespan(hour * 2)
        assert '1 day' == format_timespan(day)
        assert '2 days' == format_timespan(day * 2)
        assert '1 week' == format_timespan(week)
        assert '2 weeks' == format_timespan(week * 2)
        assert '1 year' == format_timespan(year)
        assert '2 years' == format_timespan(year * 2)
        assert '6 years, 5 weeks, 4 days, 3 hours, 2 minutes and 500 milliseconds' == format_timespan(year * 6 + week * 5 + day * 4 + hour * 3 + minute * 2 + 0.5, detailed=True)
        assert '1 year, 2 weeks and 3 days' == format_timespan(year + week * 2 + day * 3 + hour * 12)
        assert '1 minute, 1 second and 100 milliseconds' == format_timespan(61.1, detailed=True)
        assert '1 minute and 1.1 seconds' == format_timespan(61.1, detailed=False)
        assert '1 minute and 0.3 seconds' == format_timespan(60.3)
        assert '5 minutes and 0.3 seconds' == format_timespan(300.3)
        assert '1 second and 15 milliseconds' == format_timespan(1.015, detailed=True)
        assert '10 seconds and 15 milliseconds' == format_timespan(10.015, detailed=True)
        assert '1 microsecond and 50 nanoseconds' == format_timespan(1.05e-06, detailed=True)
        now = datetime.datetime.now()
        then = now - datetime.timedelta(hours=23)
        assert '23 hours' == format_timespan(now - then)

    def test_parse_timespan(self):
        """Test :func:`humanfriendly.parse_timespan()`."""
        self.assertEqual(0, parse_timespan('0'))
        self.assertEqual(0, parse_timespan('0s'))
        self.assertEqual(1e-09, parse_timespan('1ns'))
        self.assertEqual(5.1e-08, parse_timespan('51ns'))
        self.assertEqual(1e-06, parse_timespan('1us'))
        self.assertEqual(5.2e-05, parse_timespan('52us'))
        self.assertEqual(0.001, parse_timespan('1ms'))
        self.assertEqual(0.001, parse_timespan('1 millisecond'))
        self.assertEqual(0.5, parse_timespan('500 milliseconds'))
        self.assertEqual(0.5, parse_timespan('0.5 seconds'))
        self.assertEqual(5, parse_timespan('5s'))
        self.assertEqual(5, parse_timespan('5 seconds'))
        self.assertEqual(60 * 2, parse_timespan('2m'))
        self.assertEqual(60 * 2, parse_timespan('2 minutes'))
        self.assertEqual(60 * 3, parse_timespan('3 min'))
        self.assertEqual(60 * 3, parse_timespan('3 mins'))
        self.assertEqual(60 * 60 * 3, parse_timespan('3 h'))
        self.assertEqual(60 * 60 * 3, parse_timespan('3 hours'))
        self.assertEqual(60 * 60 * 24 * 4, parse_timespan('4d'))
        self.assertEqual(60 * 60 * 24 * 4, parse_timespan('4 days'))
        self.assertEqual(60 * 60 * 24 * 7 * 5, parse_timespan('5 w'))
        self.assertEqual(60 * 60 * 24 * 7 * 5, parse_timespan('5 weeks'))
        with self.assertRaises(InvalidTimespan):
            parse_timespan('1z')

    def test_parse_date(self):
        """Test :func:`humanfriendly.parse_date()`."""
        self.assertEqual((2013, 6, 17, 0, 0, 0), parse_date('2013-06-17'))
        self.assertEqual((2013, 6, 17, 2, 47, 42), parse_date('2013-06-17 02:47:42'))
        self.assertEqual((2016, 11, 30, 0, 47, 17), parse_date(u'2016-11-30 00:47:17'))
        with self.assertRaises(InvalidDate):
            parse_date('2013-06-XY')

    def test_format_size(self):
        """Test :func:`humanfriendly.format_size()`."""
        self.assertEqual('0 bytes', format_size(0))
        self.assertEqual('1 byte', format_size(1))
        self.assertEqual('42 bytes', format_size(42))
        self.assertEqual('1 KB', format_size(1000 ** 1))
        self.assertEqual('1 MB', format_size(1000 ** 2))
        self.assertEqual('1 GB', format_size(1000 ** 3))
        self.assertEqual('1 TB', format_size(1000 ** 4))
        self.assertEqual('1 PB', format_size(1000 ** 5))
        self.assertEqual('1 EB', format_size(1000 ** 6))
        self.assertEqual('1 ZB', format_size(1000 ** 7))
        self.assertEqual('1 YB', format_size(1000 ** 8))
        self.assertEqual('1 KiB', format_size(1024 ** 1, binary=True))
        self.assertEqual('1 MiB', format_size(1024 ** 2, binary=True))
        self.assertEqual('1 GiB', format_size(1024 ** 3, binary=True))
        self.assertEqual('1 TiB', format_size(1024 ** 4, binary=True))
        self.assertEqual('1 PiB', format_size(1024 ** 5, binary=True))
        self.assertEqual('1 EiB', format_size(1024 ** 6, binary=True))
        self.assertEqual('1 ZiB', format_size(1024 ** 7, binary=True))
        self.assertEqual('1 YiB', format_size(1024 ** 8, binary=True))
        self.assertEqual('45 KB', format_size(1000 * 45))
        self.assertEqual('2.9 TB', format_size(1000 ** 4 * 2.9))

    def test_parse_size(self):
        """Test :func:`humanfriendly.parse_size()`."""
        self.assertEqual(0, parse_size('0B'))
        self.assertEqual(42, parse_size('42'))
        self.assertEqual(42, parse_size('42B'))
        self.assertEqual(1000, parse_size('1k'))
        self.assertEqual(1024, parse_size('1k', binary=True))
        self.assertEqual(1000, parse_size('1 KB'))
        self.assertEqual(1000, parse_size('1 kilobyte'))
        self.assertEqual(1024, parse_size('1 kilobyte', binary=True))
        self.assertEqual(1000 ** 2 * 69, parse_size('69 MB'))
        self.assertEqual(1000 ** 3, parse_size('1 GB'))
        self.assertEqual(1000 ** 4, parse_size('1 TB'))
        self.assertEqual(1000 ** 5, parse_size('1 PB'))
        self.assertEqual(1000 ** 6, parse_size('1 EB'))
        self.assertEqual(1000 ** 7, parse_size('1 ZB'))
        self.assertEqual(1000 ** 8, parse_size('1 YB'))
        self.assertEqual(1000 ** 3 * 1.5, parse_size('1.5 GB'))
        self.assertEqual(1024 ** 8 * 1.5, parse_size('1.5 YiB'))
        with self.assertRaises(InvalidSize):
            parse_size('1q')
        with self.assertRaises(InvalidSize):
            parse_size('a')

    def test_format_length(self):
        """Test :func:`humanfriendly.format_length()`."""
        self.assertEqual('0 metres', format_length(0))
        self.assertEqual('1 metre', format_length(1))
        self.assertEqual('42 metres', format_length(42))
        self.assertEqual('1 km', format_length(1 * 1000))
        self.assertEqual('15.3 cm', format_length(0.153))
        self.assertEqual('1 cm', format_length(0.01))
        self.assertEqual('1 mm', format_length(0.001))
        self.assertEqual('1 nm', format_length(1e-09))

    def test_parse_length(self):
        """Test :func:`humanfriendly.parse_length()`."""
        self.assertEqual(0, parse_length('0m'))
        self.assertEqual(42, parse_length('42'))
        self.assertEqual(1.5, parse_length('1.5'))
        self.assertEqual(42, parse_length('42m'))
        self.assertEqual(1000, parse_length('1km'))
        self.assertEqual(0.153, parse_length('15.3 cm'))
        self.assertEqual(0.01, parse_length('1cm'))
        self.assertEqual(0.001, parse_length('1mm'))
        self.assertEqual(1e-09, parse_length('1nm'))
        with self.assertRaises(InvalidLength):
            parse_length('1z')
        with self.assertRaises(InvalidLength):
            parse_length('a')

    def test_format_number(self):
        """Test :func:`humanfriendly.format_number()`."""
        self.assertEqual('1', format_number(1))
        self.assertEqual('1.5', format_number(1.5))
        self.assertEqual('1.56', format_number(1.56789))
        self.assertEqual('1.567', format_number(1.56789, 3))
        self.assertEqual('1,000', format_number(1000))
        self.assertEqual('1,000', format_number(1000.12, 0))
        self.assertEqual('1,000,000', format_number(1000000))
        self.assertEqual('1,000,000.42', format_number(1000000.42))
        self.assertEqual('-285.67', format_number(-285.67))

    def test_round_number(self):
        """Test :func:`humanfriendly.round_number()`."""
        self.assertEqual('1', round_number(1))
        self.assertEqual('1', round_number(1.0))
        self.assertEqual('1.00', round_number(1, keep_width=True))
        self.assertEqual('3.14', round_number(3.141592653589793))

    def test_format_path(self):
        """Test :func:`humanfriendly.format_path()`."""
        friendly_path = os.path.join('~', '.vimrc')
        absolute_path = os.path.join(os.environ['HOME'], '.vimrc')
        self.assertEqual(friendly_path, format_path(absolute_path))

    def test_parse_path(self):
        """Test :func:`humanfriendly.parse_path()`."""
        friendly_path = os.path.join('~', '.vimrc')
        absolute_path = os.path.join(os.environ['HOME'], '.vimrc')
        self.assertEqual(absolute_path, parse_path(friendly_path))

    def test_pretty_tables(self):
        """Test :func:`humanfriendly.tables.format_pretty_table()`."""
        data = [['Just one column']]
        assert format_pretty_table(data) == dedent('\n            -------------------\n            | Just one column |\n            -------------------\n        ').strip()
        data = [['One', 'Two', 'Three'], ['1', '2', '3']]
        assert format_pretty_table(data) == dedent('\n            ---------------------\n            | One | Two | Three |\n            | 1   | 2   | 3     |\n            ---------------------\n        ').strip()
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        assert ansi_strip(format_pretty_table(data, column_names)) == dedent('\n            ---------------------\n            | One | Two | Three |\n            ---------------------\n            | 1   | 2   | 3     |\n            | a   | b   | c     |\n            ---------------------\n        ').strip()
        column_names = ['Just a label', 'Important numbers']
        data = [['Row one', '15'], ['Row two', '300']]
        assert ansi_strip(format_pretty_table(data, column_names)) == dedent('\n            ------------------------------------\n            | Just a label | Important numbers |\n            ------------------------------------\n            | Row one      |                15 |\n            | Row two      |               300 |\n            ------------------------------------\n        ').strip()

    def test_robust_tables(self):
        """Test :func:`humanfriendly.tables.format_robust_table()`."""
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        assert ansi_strip(format_robust_table(data, column_names)) == dedent('\n            --------\n            One: 1\n            Two: 2\n            Three: 3\n            --------\n            One: a\n            Two: b\n            Three: c\n            --------\n        ').strip()
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'Here comes a\nmulti line column!']]
        assert ansi_strip(format_robust_table(data, column_names)) == dedent('\n            ------------------\n            One: 1\n            Two: 2\n            Three: 3\n            ------------------\n            One: a\n            Two: b\n            Three:\n            Here comes a\n            multi line column!\n            ------------------\n        ').strip()

    def test_smart_tables(self):
        """Test :func:`humanfriendly.tables.format_smart_table()`."""
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        assert ansi_strip(format_smart_table(data, column_names)) == dedent('\n            ---------------------\n            | One | Two | Three |\n            ---------------------\n            | 1   | 2   | 3     |\n            | a   | b   | c     |\n            ---------------------\n        ').strip()
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'Here comes a\nmulti line column!']]
        assert ansi_strip(format_smart_table(data, column_names)) == dedent('\n            ------------------\n            One: 1\n            Two: 2\n            Three: 3\n            ------------------\n            One: a\n            Two: b\n            Three:\n            Here comes a\n            multi line column!\n            ------------------\n        ').strip()

    def test_rst_tables(self):
        """Test :func:`humanfriendly.tables.format_rst_table()`."""
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        self.assertEqual(format_rst_table(data, column_names), dedent('\n                ===  ===  =====\n                One  Two  Three\n                ===  ===  =====\n                1    2    3\n                a    b    c\n                ===  ===  =====\n            ').rstrip())
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        self.assertEqual(format_rst_table(data), dedent('\n                =  =  =\n                1  2  3\n                a  b  c\n                =  =  =\n            ').rstrip())

    def test_concatenate(self):
        """Test :func:`humanfriendly.text.concatenate()`."""
        assert concatenate([]) == ''
        assert concatenate(['one']) == 'one'
        assert concatenate(['one', 'two']) == 'one and two'
        assert concatenate(['one', 'two', 'three']) == 'one, two and three'
        assert concatenate(['one', 'two', 'three'], conjunction='or') == 'one, two or three'
        assert concatenate(['one', 'two', 'three'], serial_comma=True) == 'one, two, and three'

    def test_split(self):
        """Test :func:`humanfriendly.text.split()`."""
        from humanfriendly.text import split
        self.assertEqual(split(''), [])
        self.assertEqual(split('foo'), ['foo'])
        self.assertEqual(split('foo, bar'), ['foo', 'bar'])
        self.assertEqual(split('foo, bar, baz'), ['foo', 'bar', 'baz'])
        self.assertEqual(split('foo,bar,baz'), ['foo', 'bar', 'baz'])

    def test_timer(self):
        """Test :func:`humanfriendly.Timer`."""
        for seconds, text in ((1, '1 second'), (2, '2 seconds'), (60, '1 minute'), (60 * 2, '2 minutes'), (60 * 60, '1 hour'), (60 * 60 * 2, '2 hours'), (60 * 60 * 24, '1 day'), (60 * 60 * 24 * 2, '2 days'), (60 * 60 * 24 * 7, '1 week'), (60 * 60 * 24 * 7 * 2, '2 weeks')):
            t = Timer(time.time() - seconds)
            self.assertEqual(round_number(t.elapsed_time, keep_width=True), '%i.00' % seconds)
            self.assertEqual(str(t), text)
        t = Timer(time.time() - 2.2)
        self.assertEqual(t.rounded, '2 seconds')
        automatic_timer = Timer()
        time.sleep(1)
        self.assertEqual(normalize_timestamp(automatic_timer.elapsed_time, 0), '1.00')
        resumable_timer = Timer(resumable=True)
        for i in range(2):
            with resumable_timer:
                time.sleep(1)
        self.assertEqual(normalize_timestamp(resumable_timer.elapsed_time, 0), '2.00')
        with Timer(resumable=True) as timer:
            assert timer is not None

    def test_spinner(self):
        """Test :func:`humanfriendly.Spinner`."""
        stream = StringIO()
        spinner = Spinner(label='test spinner', total=4, stream=stream, interactive=True)
        for progress in [1, 2, 3, 4]:
            spinner.step(progress=progress)
            time.sleep(0.2)
        spinner.clear()
        output = stream.getvalue()
        output = output.replace(ANSI_SHOW_CURSOR, '').replace(ANSI_HIDE_CURSOR, '')
        lines = [line for line in output.split(ANSI_ERASE_LINE) if line]
        self.assertTrue(len(lines) > 0)
        self.assertTrue(all(('test spinner' in line for line in lines)))
        self.assertTrue(all(('%' in line for line in lines)))
        self.assertEqual(sorted(set(lines)), sorted(lines))

    def test_automatic_spinner(self):
        """
        Test :func:`humanfriendly.AutomaticSpinner`.

        There's not a lot to test about the :class:`.AutomaticSpinner` class,
        but by at least running it here we are assured that the code functions
        on all supported Python versions. :class:`.AutomaticSpinner` is built
        on top of the :class:`.Spinner` class so at least we also have the
        tests for the :class:`.Spinner` class to back us up.
        """
        with AutomaticSpinner(label='test spinner'):
            time.sleep(1)

    def test_prompt_for_choice(self):
        """Test :func:`humanfriendly.prompts.prompt_for_choice()`."""
        with self.assertRaises(ValueError):
            prompt_for_choice([])
        with open(os.devnull) as handle:
            with PatchedAttribute(sys, 'stdin', handle):
                only_option = 'only one option (shortcut)'
                assert prompt_for_choice([only_option]) == only_option
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'foo'):
            assert prompt_for_choice(['foo', 'bar']) == 'foo'
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'f'):
            assert prompt_for_choice(['foo', 'bar']) == 'foo'
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: '2'):
            assert prompt_for_choice(['foo', 'bar']) == 'bar'
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
            assert prompt_for_choice(['foo', 'bar'], default='bar') == 'bar'
        replies = ['', 'q', 'z']
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
            assert prompt_for_choice(['foo', 'bar', 'baz']) == 'baz'
        replies = ['a', 'q']
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
            assert prompt_for_choice(['foo', 'bar', 'baz', 'qux']) == 'qux'
        replies = ['42', '2']
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
            assert prompt_for_choice(['foo', 'bar', 'baz']) == 'bar'
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
            with self.assertRaises(TooManyInvalidReplies):
                prompt_for_choice(['a', 'b', 'c'])

    def test_prompt_for_confirmation(self):
        """Test :func:`humanfriendly.prompts.prompt_for_confirmation()`."""
        for reply in ('yes', 'Yes', 'YES', 'y', 'Y'):
            with PatchedAttribute(prompts, 'interactive_prompt', lambda p: reply):
                assert prompt_for_confirmation('Are you sure?') is True
        for reply in ('no', 'No', 'NO', 'n', 'N'):
            with PatchedAttribute(prompts, 'interactive_prompt', lambda p: reply):
                assert prompt_for_confirmation('Are you sure?') is False
        for default_choice in (True, False):
            with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
                assert prompt_for_confirmation('Are you sure?', default=default_choice) is default_choice
        replies = ['', 'y']
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
            with CaptureOutput(merged=True) as capturer:
                assert prompt_for_confirmation('Are you sure?') is True
                assert "there's no default choice" in capturer.get_text()
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'y'):
            for default_value, expected_text in ((True, 'Y/n'), (False, 'y/N'), (None, 'y/n')):
                with CaptureOutput(merged=True) as capturer:
                    assert prompt_for_confirmation('Are you sure?', default=default_value) is True
                    assert expected_text in capturer.get_text()
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
            with self.assertRaises(TooManyInvalidReplies):
                prompt_for_confirmation('Are you sure?')

    def test_prompt_for_input(self):
        """Test :func:`humanfriendly.prompts.prompt_for_input()`."""
        with open(os.devnull) as handle:
            with PatchedAttribute(sys, 'stdin', handle):
                default_value = 'To seek the holy grail!'
                assert prompt_for_input('What is your quest?', default=default_value) == default_value
                with self.assertRaises(EOFError):
                    prompt_for_input('What is your favorite color?')

    def test_cli(self):
        """Test the command line interface."""
        returncode, output = run_cli(main)
        assert 'Usage:' in output
        returncode, output = run_cli(main, '--help')
        assert 'Usage:' in output
        returncode, output = run_cli(main, '--unsupported-option')
        assert returncode != 0
        returncode, output = run_cli(main, '--format-number=1234567')
        assert output.strip() == '1,234,567'
        random_byte_count = random.randint(1024, 1024 * 1024)
        returncode, output = run_cli(main, '--format-size=%i' % random_byte_count)
        assert output.strip() == format_size(random_byte_count)
        random_byte_count = random.randint(1024, 1024 * 1024)
        returncode, output = run_cli(main, '--format-size=%i' % random_byte_count, '--binary')
        assert output.strip() == format_size(random_byte_count, binary=True)
        random_len = random.randint(1024, 1024 * 1024)
        returncode, output = run_cli(main, '--format-length=%i' % random_len)
        assert output.strip() == format_length(random_len)
        random_len = float(random_len) / 12345.6
        returncode, output = run_cli(main, '--format-length=%f' % random_len)
        assert output.strip() == format_length(random_len)
        returncode, output = run_cli(main, '--format-table', '--delimiter=\t', input='1\t2\t3\n4\t5\t6\n7\t8\t9')
        assert output.strip() == dedent('\n            -------------\n            | 1 | 2 | 3 |\n            | 4 | 5 | 6 |\n            | 7 | 8 | 9 |\n            -------------\n        ').strip()
        random_timespan = random.randint(5, 600)
        returncode, output = run_cli(main, '--format-timespan=%i' % random_timespan)
        assert output.strip() == format_timespan(random_timespan)
        returncode, output = run_cli(main, '--parse-size=5 KB')
        assert int(output) == parse_size('5 KB')
        returncode, output = run_cli(main, '--parse-size=5 YiB')
        assert int(output) == parse_size('5 YB', binary=True)
        returncode, output = run_cli(main, '--parse-length=5 km')
        assert int(output) == parse_length('5 km')
        returncode, output = run_cli(main, '--parse-length=1.05 km')
        assert float(output) == parse_length('1.05 km')
        returncode, output = run_cli(main, '--run-command', 'bash', '-c', 'sleep 2 && exit 42')
        assert returncode == 42
        returncode, output = run_cli(main, '--demo')
        assert returncode == 0
        lines = [ansi_strip(line) for line in output.splitlines()]
        assert 'Text styles:' in lines
        assert 'Foreground colors:' in lines
        assert 'Background colors:' in lines
        assert '256 color mode (standard colors):' in lines
        assert '256 color mode (high-intensity colors):' in lines
        assert '256 color mode (216 colors):' in lines
        assert '256 color mode (gray scale colors):' in lines

    def test_ansi_style(self):
        """Test :func:`humanfriendly.terminal.ansi_style()`."""
        assert ansi_style(bold=True) == '%s1%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(faint=True) == '%s2%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(italic=True) == '%s3%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(underline=True) == '%s4%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(inverse=True) == '%s7%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(strike_through=True) == '%s9%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color='blue') == '%s34%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(background='blue') == '%s44%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color='blue', bright=True) == '%s94%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color=214) == '%s38;5;214%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(background=214) == '%s39;5;214%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color=(0, 0, 0)) == '%s38;2;0;0;0%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color=(255, 255, 255)) == '%s38;2;255;255;255%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(background=(50, 100, 150)) == '%s48;2;50;100;150%s' % (ANSI_CSI, ANSI_SGR)
        with self.assertRaises(ValueError):
            ansi_style(color='unknown')

    def test_ansi_width(self):
        """Test :func:`humanfriendly.terminal.ansi_width()`."""
        text = 'Whatever'
        assert len(text) == ansi_width(text)
        wrapped = ansi_wrap(text, bold=True)
        assert wrapped != text
        assert len(wrapped) > len(text)
        assert len(text) == ansi_width(wrapped)

    def test_ansi_wrap(self):
        """Test :func:`humanfriendly.terminal.ansi_wrap()`."""
        text = 'Whatever'
        assert text == ansi_wrap(text)
        assert ansi_wrap(text, bold=True).startswith(ANSI_CSI)
        assert ansi_wrap(text, bold=True).endswith(ANSI_RESET)

    def test_html_to_ansi(self):
        """Test the :func:`humanfriendly.terminal.html_to_ansi()` function."""
        assert html_to_ansi('Just some plain text') == 'Just some plain text'
        assert html_to_ansi('<a href="https://python.org">python.org</a>') == '\x1b[0m\x1b[4;94mpython.org\x1b[0m (\x1b[0m\x1b[4;94mhttps://python.org\x1b[0m)'
        assert html_to_ansi('<a href="mailto:peter@peterodding.com">peter@peterodding.com</a>') == '\x1b[0m\x1b[4;94mpeter@peterodding.com\x1b[0m'
        assert html_to_ansi("Let's try <b>bold</b>") == "Let's try \x1b[0m\x1b[1mbold\x1b[0m"
        assert html_to_ansi('Let\'s try <span style="font-weight: bold">bold</span>') == "Let's try \x1b[0m\x1b[1mbold\x1b[0m"
        assert html_to_ansi("Let's try <i>italic</i>") == "Let's try \x1b[0m\x1b[3mitalic\x1b[0m"
        assert html_to_ansi('Let\'s try <span style="font-style: italic">italic</span>') == "Let's try \x1b[0m\x1b[3mitalic\x1b[0m"
        assert html_to_ansi("Let's try <ins>underline</ins>") == "Let's try \x1b[0m\x1b[4munderline\x1b[0m"
        assert html_to_ansi('Let\'s try <span style="text-decoration: underline">underline</span>') == "Let's try \x1b[0m\x1b[4munderline\x1b[0m"
        assert html_to_ansi("Let's try <s>strike-through</s>") == "Let's try \x1b[0m\x1b[9mstrike-through\x1b[0m"
        assert html_to_ansi('Let\'s try <span style="text-decoration: line-through">strike-through</span>') == "Let's try \x1b[0m\x1b[9mstrike-through\x1b[0m"
        assert html_to_ansi("Let's try <code>pre-formatted</code>") == "Let's try \x1b[0m\x1b[33mpre-formatted\x1b[0m"
        assert html_to_ansi('Let\'s try <span style="color: #AABBCC">text colors</s>') == "Let's try \x1b[0m\x1b[38;2;170;187;204mtext colors\x1b[0m"
        assert html_to_ansi('Let\'s try <span style="background-color: rgb(50, 50, 50)">background colors</s>') == "Let's try \x1b[0m\x1b[48;2;50;50;50mbackground colors\x1b[0m"
        assert html_to_ansi("Let's try some<br>line<br>breaks") == "Let's try some\nline\nbreaks"
        assert html_to_ansi('&#38;') == '&'
        assert html_to_ansi('&amp;') == '&'
        assert html_to_ansi('&gt;') == '>'
        assert html_to_ansi('&lt;') == '<'
        assert html_to_ansi('&#x26;') == '&'

        def callback(text):
            return text.replace(':wink:', ';-)')
        assert ':wink:' not in html_to_ansi('<b>:wink:</b>', callback=callback)
        assert ':wink:' in html_to_ansi('<code>:wink:</code>', callback=callback)
        assert html_to_ansi(u'\n            Tweakers zit er idd nog steeds:<br><br>\n            peter@peter-work&gt; curl -s <a href="tweakers.net">tweakers.net</a> | grep -i hosting<br>\n            &lt;a href="<a href="http://www.true.nl/webhosting/">http://www.true.nl/webhosting/</a>"\n                rel="external" id="true" title="Hosting door True"&gt;&lt;/a&gt;<br>\n            Hosting door &lt;a href="<a href="http://www.true.nl/vps/">http://www.true.nl/vps/</a>"\n                title="VPS hosting" rel="external"&gt;True</a>\n        ')

    def test_generate_output(self):
        """Test the :func:`humanfriendly.terminal.output()` function."""
        text = 'Standard output generated by output()'
        with CaptureOutput(merged=False) as capturer:
            output(text)
            self.assertEqual([text], capturer.stdout.get_lines())
            self.assertEqual([], capturer.stderr.get_lines())

    def test_generate_message(self):
        """Test the :func:`humanfriendly.terminal.message()` function."""
        text = 'Standard error generated by message()'
        with CaptureOutput(merged=False) as capturer:
            message(text)
            self.assertEqual([], capturer.stdout.get_lines())
            self.assertEqual([text], capturer.stderr.get_lines())

    def test_generate_warning(self):
        """Test the :func:`humanfriendly.terminal.warning()` function."""
        from capturer import CaptureOutput
        text = 'Standard error generated by warning()'
        with CaptureOutput(merged=False) as capturer:
            warning(text)
            self.assertEqual([], capturer.stdout.get_lines())
            self.assertEqual([ansi_wrap(text, color='red')], self.ignore_coverage_warning(capturer.stderr))

    def ignore_coverage_warning(self, stream):
        """
        Filter out coverage.py warning from standard error.

        This is intended to remove the following line from the lines captured
        on the standard error stream:

        Coverage.py warning: No data was collected. (no-data-collected)
        """
        return [line for line in stream.get_lines() if 'no-data-collected' not in line]

    def test_clean_output(self):
        """Test :func:`humanfriendly.terminal.clean_terminal_output()`."""
        assert clean_terminal_output('foo') == ['foo']
        assert clean_terminal_output('foo\nbar') == ['foo', 'bar']
        assert clean_terminal_output('foo\rbar\nbaz') == ['bar', 'baz']
        assert clean_terminal_output('aaa\rab') == ['aba']
        assert clean_terminal_output('aaa\x08\x08b') == ['aba']
        assert clean_terminal_output('foo\nbar\nbaz\n\n\n') == ['foo', 'bar', 'baz']

    def test_find_terminal_size(self):
        """Test :func:`humanfriendly.terminal.find_terminal_size()`."""
        lines, columns = find_terminal_size()
        assert lines > 0
        assert columns > 0
        saved_stdin = sys.stdin
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
            sys.stdin = StringIO()
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            lines, columns = find_terminal_size()
            assert lines > 0
            assert columns > 0
            saved_path = os.environ['PATH']
            try:
                os.environ['PATH'] = ''
                lines, columns = find_terminal_size()
                assert lines > 0
                assert columns > 0
            finally:
                os.environ['PATH'] = saved_path
        finally:
            sys.stdin = saved_stdin
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr

    def test_terminal_capabilities(self):
        """Test the functions that check for terminal capabilities."""
        from capturer import CaptureOutput
        for test_stream in (connected_to_terminal, terminal_supports_colors):
            for stream in (sys.stdout, sys.stderr):
                with CaptureOutput():
                    assert test_stream(stream)
            with open(os.devnull) as handle:
                assert not test_stream(handle)
            assert not test_stream(object())

    def test_show_pager(self):
        """Test :func:`humanfriendly.terminal.show_pager()`."""
        original_pager = os.environ.get('PAGER', None)
        try:
            os.environ['PAGER'] = 'cat'
            random_text = '\n'.join((random_string(25) for i in range(50)))
            with CaptureOutput() as capturer:
                show_pager(random_text)
                assert random_text in capturer.get_text()
        finally:
            if original_pager is not None:
                os.environ['PAGER'] = original_pager
            else:
                os.environ.pop('PAGER')

    def test_get_pager_command(self):
        """Test :func:`humanfriendly.terminal.get_pager_command()`."""
        assert '--RAW-CONTROL-CHARS' not in get_pager_command('Usage message')
        assert '--RAW-CONTROL-CHARS' in get_pager_command(ansi_wrap('Usage message', bold=True))
        options_specific_to_less = ['--no-init', '--quit-if-one-screen']
        for pager in ('cat', 'less'):
            original_pager = os.environ.get('PAGER', None)
            try:
                os.environ['PAGER'] = pager
                command_line = get_pager_command()
                if pager == 'less':
                    assert all((opt in command_line for opt in options_specific_to_less))
                else:
                    assert not any((opt in command_line for opt in options_specific_to_less))
            finally:
                if original_pager is not None:
                    os.environ['PAGER'] = original_pager
                else:
                    os.environ.pop('PAGER')

    def test_find_meta_variables(self):
        """Test :func:`humanfriendly.usage.find_meta_variables()`."""
        assert sorted(find_meta_variables("\n            Here's one example: --format-number=VALUE\n            Here's another example: --format-size=BYTES\n            A final example: --format-timespan=SECONDS\n            This line doesn't contain a META variable.\n        ")) == sorted(['VALUE', 'BYTES', 'SECONDS'])

    def test_parse_usage_simple(self):
        """Test :func:`humanfriendly.usage.parse_usage()` (a simple case)."""
        introduction, options = self.preprocess_parse_result('\n            Usage: my-fancy-app [OPTIONS]\n\n            Boring description.\n\n            Supported options:\n\n              -h, --help\n\n                Show this message and exit.\n        ')
        assert 'Usage: my-fancy-app [OPTIONS]' in introduction
        assert 'Boring description.' in introduction
        assert 'Supported options:' in introduction
        assert '-h, --help' in options
        assert 'Show this message and exit.' in options

    def test_parse_usage_tricky(self):
        """Test :func:`humanfriendly.usage.parse_usage()` (a tricky case)."""
        introduction, options = self.preprocess_parse_result("\n            Usage: my-fancy-app [OPTIONS]\n\n            Here's the introduction to my-fancy-app. Some of the lines in the\n            introduction start with a command line option just to confuse the\n            parsing algorithm :-)\n\n            For example\n            --an-awesome-option\n            is still part of the introduction.\n\n            Supported options:\n\n              -a, --an-awesome-option\n\n                Explanation why this is an awesome option.\n\n              -b, --a-boring-option\n\n                Explanation why this is a boring option.\n        ")
        assert 'Usage: my-fancy-app [OPTIONS]' in introduction
        assert any(('still part of the introduction' in p for p in introduction))
        assert 'Supported options:' in introduction
        assert '-a, --an-awesome-option' in options
        assert 'Explanation why this is an awesome option.' in options
        assert '-b, --a-boring-option' in options
        assert 'Explanation why this is a boring option.' in options

    def test_parse_usage_commas(self):
        """Test :func:`humanfriendly.usage.parse_usage()` against option labels containing commas."""
        introduction, options = self.preprocess_parse_result("\n            Usage: my-fancy-app [OPTIONS]\n\n            Some introduction goes here.\n\n            Supported options:\n\n              -f, --first-option\n\n                Explanation of first option.\n\n              -s, --second-option=WITH,COMMA\n\n                This should be a separate option's description.\n        ")
        assert 'Usage: my-fancy-app [OPTIONS]' in introduction
        assert 'Some introduction goes here.' in introduction
        assert 'Supported options:' in introduction
        assert '-f, --first-option' in options
        assert 'Explanation of first option.' in options
        assert '-s, --second-option=WITH,COMMA' in options
        assert "This should be a separate option's description." in options

    def preprocess_parse_result(self, text):
        """Ignore leading/trailing whitespace in usage parsing tests."""
        return tuple(([p.strip() for p in r] for r in parse_usage(dedent(text))))

    def test_format_usage(self):
        """Test :func:`humanfriendly.usage.format_usage()`."""
        usage_text = 'Just one --option'
        formatted_text = format_usage(usage_text)
        assert len(formatted_text) > len(usage_text)
        assert formatted_text.startswith('Just one ')
        usage_text = 'Usage: humanfriendly [OPTIONS]'
        formatted_text = format_usage(usage_text)
        assert len(formatted_text) > len(usage_text)
        assert usage_text in formatted_text
        assert not formatted_text.startswith(usage_text)
        usage_text = '--valid-option=VALID_METAVAR\nVALID_METAVAR is bogus\nINVALID_METAVAR should not be highlighted\n'
        formatted_text = format_usage(usage_text)
        formatted_lines = formatted_text.splitlines()
        assert ANSI_CSI in formatted_lines[1]
        assert ANSI_CSI not in formatted_lines[2]

    def test_render_usage(self):
        """Test :func:`humanfriendly.usage.render_usage()`."""
        assert render_usage('Usage: some-command WITH ARGS') == '**Usage:** `some-command WITH ARGS`'
        assert render_usage('Supported options:') == '**Supported options:**'
        assert 'code-block' in render_usage(dedent('\n            Here comes a shell command:\n\n              $ echo test\n              test\n        '))
        assert all((token in render_usage(dedent("\n            Supported options:\n\n              -n, --dry-run\n\n                Don't change anything.\n        ")) for token in ('`-n`', '`--dry-run`')))

    def test_deprecated_args(self):
        """Test the deprecated_args() decorator function."""

        @deprecated_args('foo', 'bar')
        def test_function(**options):
            assert options['foo'] == 'foo'
            assert options.get('bar') in (None, 'bar')
            return 42
        fake_fn = MagicMock()
        with PatchedAttribute(warnings, 'warn', fake_fn):
            assert test_function('foo', 'bar') == 42
            with self.assertRaises(TypeError):
                test_function('foo', 'bar', 'baz')
        assert fake_fn.was_called

    def test_alias_proxy_deprecation_warning(self):
        """Test that the DeprecationProxy class emits deprecation warnings."""
        fake_fn = MagicMock()
        with PatchedAttribute(warnings, 'warn', fake_fn):
            module = sys.modules[__name__]
            aliases = dict(concatenate='humanfriendly.text.concatenate')
            proxy = DeprecationProxy(module, aliases)
            assert proxy.concatenate == concatenate
        assert fake_fn.was_called

    def test_alias_proxy_sphinx_compensation(self):
        """Test that the DeprecationProxy class emits deprecation warnings."""
        with PatchedItem(sys.modules, 'sphinx', types.ModuleType('sphinx')):
            define_aliases(__name__, concatenate='humanfriendly.text.concatenate')
            assert 'concatenate' in dir(sys.modules[__name__])
            assert 'concatenate' in get_aliases(__name__)

    def test_alias_proxy_sphinx_integration(self):
        """Test that aliases can be injected into generated documentation."""
        module = sys.modules[__name__]
        define_aliases(__name__, concatenate='humanfriendly.text.concatenate')
        lines = module.__doc__.splitlines()
        deprecation_note_callback(app=None, what=None, name=None, obj=module, options=None, lines=lines)
        assert '\n'.join(lines) != module.__doc__

    def test_sphinx_customizations(self):
        """Test the :mod:`humanfriendly.sphinx` module."""

        class FakeApp(object):

            def __init__(self):
                self.callbacks = {}
                self.roles = {}

            def __documented_special_method__(self):
                """Documented unofficial special method."""
                pass

            def __undocumented_special_method__(self):
                pass

            def add_role(self, name, callback):
                self.roles[name] = callback

            def connect(self, event, callback):
                self.callbacks.setdefault(event, []).append(callback)

            def bogus_usage(self):
                """Usage: This is not supposed to be reformatted!"""
                pass
        fake_app = FakeApp()
        setup(fake_app)
        assert man_role == fake_app.roles['man']
        assert pypi_role == fake_app.roles['pypi']
        assert deprecation_note_callback in fake_app.callbacks['autodoc-process-docstring']
        assert special_methods_callback in fake_app.callbacks['autodoc-skip-member']
        assert usage_message_callback in fake_app.callbacks['autodoc-process-docstring']
        assert special_methods_callback(app=None, what=None, name=None, obj=FakeApp.__documented_special_method__, skip=True, options=None) is False
        assert special_methods_callback(app=None, what=None, name=None, obj=FakeApp.__undocumented_special_method__, skip=True, options=None) is True
        from humanfriendly import cli, sphinx
        assert self.docstring_is_reformatted(cli)
        assert not self.docstring_is_reformatted(sphinx)
        assert not self.docstring_is_reformatted(fake_app.bogus_usage)

    def docstring_is_reformatted(self, entity):
        """Check whether :func:`.usage_message_callback()` reformats a module's docstring."""
        lines = trim_empty_lines(entity.__doc__).splitlines()
        saved_lines = list(lines)
        usage_message_callback(app=None, what=None, name=None, obj=entity, options=None, lines=lines)
        return lines != saved_lines