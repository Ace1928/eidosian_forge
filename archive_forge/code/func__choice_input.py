import datetime
import importlib
import os
import sys
from django.apps import apps
from django.core.management.base import OutputWrapper
from django.db.models import NOT_PROVIDED
from django.utils import timezone
from django.utils.version import get_docs_version
from .loader import MigrationLoader
def _choice_input(self, question, choices):
    self.prompt_output.write(f'{question}')
    for i, choice in enumerate(choices):
        self.prompt_output.write(' %s) %s' % (i + 1, choice))
    self.prompt_output.write('Select an option: ', ending='')
    result = input()
    while True:
        try:
            value = int(result)
        except ValueError:
            pass
        else:
            if 0 < value <= len(choices):
                return value
        self.prompt_output.write('Please select a valid option: ', ending='')
        result = input()