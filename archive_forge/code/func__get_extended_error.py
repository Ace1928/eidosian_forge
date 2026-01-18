from __future__ import (absolute_import, division, print_function)
import re
import traceback
from collections.abc import Sequence
from ansible.errors.yaml_strings import (
from ansible.module_utils.common.text.converters import to_native, to_text
def _get_extended_error(self):
    """
        Given an object reporting the location of the exception in a file, return
        detailed information regarding it including:

          * the line which caused the error as well as the one preceding it
          * causes and suggested remedies for common syntax errors

        If this error was created with show_content=False, the reporting of content
        is suppressed, as the file contents may be sensitive (ie. vault data).
        """
    error_message = ''
    try:
        src_file, line_number, col_number = self.obj.ansible_pos
        error_message += YAML_POSITION_DETAILS % (src_file, line_number, col_number)
        if src_file not in ('<string>', '<unicode>') and self._show_content:
            target_line, prev_line = self._get_error_lines_from_file(src_file, line_number - 1)
            target_line = to_text(target_line)
            prev_line = to_text(prev_line)
            if target_line:
                stripped_line = target_line.replace(' ', '')
                if re.search('\\w+(\\s+)?=(\\s+)?[\\w/-]+', prev_line):
                    error_position = prev_line.rstrip().find('=')
                    arrow_line = ' ' * error_position + '^ here'
                    error_message = YAML_POSITION_DETAILS % (src_file, line_number - 1, error_position + 1)
                    error_message += '\nThe offending line appears to be:\n\n%s\n%s\n\n' % (prev_line.rstrip(), arrow_line)
                    error_message += YAML_AND_SHORTHAND_ERROR
                else:
                    arrow_line = ' ' * (col_number - 1) + '^ here'
                    error_message += '\nThe offending line appears to be:\n\n%s\n%s\n%s\n' % (prev_line.rstrip(), target_line.rstrip(), arrow_line)
                if '\t' in target_line:
                    error_message += YAML_COMMON_LEADING_TAB_ERROR
                if ('{{' in target_line and '}}' in target_line) and ('"{{' not in target_line or "'{{" not in target_line):
                    error_message += YAML_COMMON_UNQUOTED_VARIABLE_ERROR
                elif ':{{' in stripped_line and '}}' in stripped_line:
                    error_message += YAML_COMMON_DICT_ERROR
                elif len(target_line) and len(target_line) > 1 and (len(target_line) > col_number) and (target_line[col_number] == ':') and (target_line.count(':') > 1):
                    error_message += YAML_COMMON_UNQUOTED_COLON_ERROR
                else:
                    parts = target_line.split(':')
                    if len(parts) > 1:
                        middle = parts[1].strip()
                        match = False
                        unbalanced = False
                        if middle.startswith("'") and (not middle.endswith("'")):
                            match = True
                        elif middle.startswith('"') and (not middle.endswith('"')):
                            match = True
                        if len(middle) > 0 and middle[0] in ['"', "'"] and (middle[-1] in ['"', "'"]) and (target_line.count("'") > 2) or target_line.count('"') > 2:
                            unbalanced = True
                        if match:
                            error_message += YAML_COMMON_PARTIALLY_QUOTED_LINE_ERROR
                        if unbalanced:
                            error_message += YAML_COMMON_UNBALANCED_QUOTES_ERROR
    except (IOError, TypeError):
        error_message += '\n(could not open file to display line)'
    except IndexError:
        error_message += '\n(specified line no longer in file, maybe it changed?)'
    return error_message