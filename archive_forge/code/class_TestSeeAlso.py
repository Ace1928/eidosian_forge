import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class TestSeeAlso(tests.TestCase):
    """Tests for the see also functional of Command."""

    @staticmethod
    def _get_command_with_see_also(see_also):

        class ACommand(commands.Command):
            __doc__ = 'A sample command.'
            _see_also = see_also
        return ACommand()

    def test_default_subclass_no_see_also(self):
        command = self._get_command_with_see_also([])
        self.assertEqual([], command.get_see_also())

    def test__see_also(self):
        """When _see_also is defined, it sets the result of get_see_also()."""
        command = self._get_command_with_see_also(['bar', 'foo'])
        self.assertEqual(['bar', 'foo'], command.get_see_also())

    def test_deduplication(self):
        """Duplicates in _see_also are stripped out."""
        command = self._get_command_with_see_also(['foo', 'foo'])
        self.assertEqual(['foo'], command.get_see_also())

    def test_sorted(self):
        """_see_also is sorted by get_see_also."""
        command = self._get_command_with_see_also(['foo', 'bar'])
        self.assertEqual(['bar', 'foo'], command.get_see_also())

    def test_additional_terms(self):
        """Additional terms can be supplied and are deduped and sorted."""
        command = self._get_command_with_see_also(['foo', 'bar'])
        self.assertEqual(['bar', 'foo', 'gam'], command.get_see_also(['gam', 'bar', 'gam']))