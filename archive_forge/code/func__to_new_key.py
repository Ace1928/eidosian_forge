import codecs
import sys
import click
from celery.app import defaults
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
from celery.utils.functional import pass1
def _to_new_key(line, keyfilter=pass1, source=defaults._TO_NEW_KEY):
    for old_key in reversed(sorted(source, key=lambda x: len(x))):
        new_line = line.replace(old_key, keyfilter(source[old_key]))
        if line != new_line and 'CELERY_CELERY' not in new_line:
            return (1, new_line)
    return (0, line)