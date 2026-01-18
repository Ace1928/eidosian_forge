from functools import partial
from typing import Literal
import click
from kombu.utils.json import dumps
from celery.bin.base import COMMA_SEPARATED_LIST, CeleryCommand, CeleryOption, handle_preload_options
from celery.exceptions import CeleryCommandException
from celery.platforms import EX_UNAVAILABLE
from celery.utils import text
from celery.worker.control import Panel
def _compile_arguments(command, args):
    meta = Panel.meta[command]
    arguments = {}
    if meta.args:
        arguments.update({k: v for k, v in _consume_arguments(meta, command, args)})
    if meta.variadic:
        arguments.update({meta.variadic: args})
    return arguments