from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from gslib.exception import CommandException
def SanityCheck(help_provider, help_name_map):
    """Helper for checking that a HelpProvider has minimally adequate content."""
    help_name_len = len(help_provider.help_spec.help_name)
    assert help_name_len > 1 and help_name_len < MAX_HELP_NAME_LEN, 'The help name "{text}" must be less then {max}'.format(text=help_provider.help_spec.help_name, max=MAX_HELP_NAME_LEN)
    for hna in help_provider.help_spec.help_name_aliases:
        assert hna
    one_line_summary_len = len(help_provider.help_spec.help_one_line_summary)
    assert one_line_summary_len >= MIN_ONE_LINE_SUMMARY_LEN, ('The one line summary "{text}" with a length of {length} must be ' + 'more then {min} characters').format(text=help_provider.help_spec.help_one_line_summary, length=one_line_summary_len, min=MIN_ONE_LINE_SUMMARY_LEN)
    assert one_line_summary_len <= MAX_ONE_LINE_SUMMARY_LEN, ('The one line summary "{text}" with a length of {length} must be ' + 'less then {max} characters').format(text=help_provider.help_spec.help_one_line_summary, length=one_line_summary_len, max=MAX_ONE_LINE_SUMMARY_LEN)
    assert len(help_provider.help_spec.help_text) > 10, 'The length of "{text}" must be less then 10'.format(text=help_provider.help_spec.help_text)
    name_check_list = [help_provider.help_spec.help_name]
    name_check_list.extend(help_provider.help_spec.help_name_aliases)
    for name_or_alias in name_check_list:
        if name_or_alias in help_name_map:
            raise CommandException('Duplicate help name/alias "%s" found while loading help from %s. That name/alias was already taken by %s' % (name_or_alias, help_provider.__module__, help_name_map[name_or_alias].__module__))