from parlai.core.teachers import DialogTeacher
from .build import build
import os
import unicodedata
def _fix_missing_period(line):
    """
    Adds a period to a line that is missing a period.
    """
    dm_single_close_quote = u'’'
    dm_double_close_quote = u'”'
    END_TOKENS = ['.', '!', '?', '...', "'", '`', '"', dm_single_close_quote, dm_double_close_quote, ')']
    if '@highlight' in line or line == '' or line[-1] in END_TOKENS:
        return line
    return line + '.'