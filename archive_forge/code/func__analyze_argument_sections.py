from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def _analyze_argument_sections(self, heading, section):
    """Raise violation if the section contains unmatched double backticks.

    This check confirms that argument sections follow our help text style guide.
    The help text for individual arguments should not begin with an article.
    See go/cloud-sdk-help-text#formatting.

    Arguments:
      heading: str, the name of the section.
      section: str, the contents of the section.

    Returns:
      None.
    """
    has_errors = self.check_for_personal_pronouns(heading, section) or self.check_for_articles(heading, section)
    check_name = self._check_name(heading, 'ARG_ARTICLES')
    flags_with_articles = []
    all_lines_in_section = section.split('\n')
    non_empty_lines_in_section = [line.strip() for line in all_lines_in_section if not line.isspace() and line]
    prev_line = ''
    for line in non_empty_lines_in_section:
        if prev_line and (prev_line.startswith('--') or re.match('[A-Z_]', prev_line.split()[0])) and (len(prev_line.split()) < 5) and (line.split()[0].lower() in self._ARTICLES):
            flags_with_articles.append(prev_line)
        prev_line = line
    if flags_with_articles:
        has_errors = True
        self._add_failure(check_name, 'Please fix the help text for the following arguments which begin with an article in the {} section:\n{}'.format(heading, '\n'.join(flags_with_articles)))
    else:
        self._add_success(check_name)
    if not has_errors:
        self._add_no_errors_summary(heading)