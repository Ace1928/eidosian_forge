import os
import signal
import subprocess
import time
from wandb_watchdog.utils import echo, has_attribute
from wandb_watchdog.events import PatternMatchingEventHandler
class Trick(PatternMatchingEventHandler):
    """Your tricks should subclass this class."""

    @classmethod
    def generate_yaml(cls):
        context = dict(module_name=cls.__module__, klass_name=cls.__name__)
        template_yaml = '- %(module_name)s.%(klass_name)s:\n  args:\n  - argument1\n  - argument2\n  kwargs:\n    patterns:\n    - "*.py"\n    - "*.js"\n    ignore_patterns:\n    - "version.py"\n    ignore_directories: false\n'
        return template_yaml % context