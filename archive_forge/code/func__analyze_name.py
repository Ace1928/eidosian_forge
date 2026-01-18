from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def _analyze_name(self, heading, section):
    has_errors = self.check_for_personal_pronouns(heading, section) or self.check_for_articles(heading, section)
    section_parts = re.split('\\s-\\s?', section.strip())
    check_name = self._check_name('NAME', 'DESCRIPTION')
    if len(section_parts) == 1 or (len(section_parts) > 1 and (not section_parts[1].strip())):
        self.name_section = ''
        self._add_failure(check_name, 'Please add an explanation for the command.')
        has_errors = True
    else:
        self.name_section = section_parts[1]
        self._add_success(check_name)
    check_name = self._check_name('NAME', 'LENGTH')
    self.command_name = ' '.join(section_parts[0].strip().split())
    self.command_name_length = len(self.command_name)
    if len(self.name_section.split()) > self._NAME_WORD_LIMIT:
        self._add_failure(check_name, 'Please shorten the name section description to less than {} words.'.format(six.text_type(self._NAME_WORD_LIMIT)))
        has_errors = True
    else:
        self._add_success(check_name)
    if not has_errors:
        self._add_no_errors_summary(heading)