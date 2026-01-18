import collections
import copy
import importlib.metadata
import json
import logging
import operator
import sys
import yaml
from oslo_config import cfg
from oslo_i18n import _message
import stevedore.named  # noqa
def _get_choice_text(self, choice):
    if choice is None:
        return '<None>'
    elif choice == '':
        return "''"
    return str(choice)