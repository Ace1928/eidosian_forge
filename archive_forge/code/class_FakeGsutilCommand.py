from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from unittest import mock
from gslib import command
from gslib.tests import testcase
from gslib.utils import constants
class FakeGsutilCommand(command.Command):
    """Implementation of a fake gsutil command."""
    command_spec = command.Command.CreateCommandSpec('fake_gsutil', min_args=1, max_args=constants.NO_MAX, supported_sub_args='rz:', file_url_ok=True)
    help_spec = command.Command.HelpSpec(help_name='fake_gsutil', help_name_aliases=[], help_type='command_help', help_one_line_summary='Fake one line summary for the command.', help_text='Help text for fake command.', subcommand_help_text={})