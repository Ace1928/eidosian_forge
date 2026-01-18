from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def _analyze_examples(self, heading, section):
    self._has_example_section = True
    has_errors = self.check_for_articles(heading, section)
    if not self.command_metadata.is_group:
        if self.check_for_personal_pronouns(heading, section):
            has_errors = True
        if self.check_for_unmatched_double_backticks(heading, section):
            has_errors = True
        check_name = self._check_name(heading, 'FLAG_EQUALS')
        if self.equals_violation_flags:
            has_errors = True
            list_contents = ''
            for flag in range(len(self.equals_violation_flags) - 1):
                list_contents += six.text_type(self.equals_violation_flags[flag]) + ', '
            list_contents += six.text_type(self.equals_violation_flags[-1])
            self._add_failure(check_name, 'There should be an `=` between the flag name and the value for the following flags: {}'.format(list_contents))
            has_errors = True
        else:
            self._add_success(check_name)
        check_name = self._check_name(heading, 'NONEXISTENT_FLAG')
        if self.nonexistent_violation_flags:
            has_errors = True
            list_contents = ''
            for flag in range(len(self.nonexistent_violation_flags) - 1):
                list_contents += six.text_type(self.nonexistent_violation_flags[flag]) + ', '
            list_contents += six.text_type(self.nonexistent_violation_flags[-1])
            self._add_failure(check_name, 'The following flags are not valid: {}'.format(list_contents))
        else:
            self._add_success(check_name)
        self._example_errors = has_errors