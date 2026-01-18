import re
from collections import Counter
from fileinput import FileInput
import click
from celery.bin.base import CeleryCommand, handle_preload_options
class _task_counts(list):

    @property
    def format(self):
        return '\n'.join(('{}: {}'.format(*i) for i in self))