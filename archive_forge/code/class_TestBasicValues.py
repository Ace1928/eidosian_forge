import argparse
import functools
from cliff import command
from cliff.tests import base
class TestBasicValues(base.TestBase):

    def test_get_parser(self):
        cmd = TestCommand(None, None)
        parser = cmd.get_parser('NAME')
        assert parser.prog == 'NAME'

    def test_get_name(self):
        cmd = TestCommand(None, None, cmd_name='object action')
        assert cmd.cmd_name == 'object action'

    def test_run_return(self):
        cmd = TestCommand(None, None, cmd_name='object action')
        assert cmd.run(None) == 42