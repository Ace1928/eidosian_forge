import argparse
import logging
import os
import pathlib
import re
def _clean_line(line, names):
    if names:
        joined_names = '|'.join(names)
        line = re.sub(f'\\b({joined_names})\\.', '', line)
    return _USELESS_ASSIGNMENT_RE.sub('', line)