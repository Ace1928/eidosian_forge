from functools import partial
from typing import Literal
import click
from kombu.utils.json import dumps
from celery.bin.base import COMMA_SEPARATED_LIST, CeleryCommand, CeleryOption, handle_preload_options
from celery.exceptions import CeleryCommandException
from celery.platforms import EX_UNAVAILABLE
from celery.utils import text
from celery.worker.control import Panel
def _consume_arguments(meta, method, args):
    i = 0
    try:
        for i, arg in enumerate(args):
            try:
                name, typ = meta.args[i]
            except IndexError:
                if meta.variadic:
                    break
                raise click.UsageError('Command {!r} takes arguments: {}'.format(method, meta.signature))
            else:
                yield (name, typ(arg) if typ is not None else arg)
    finally:
        args[:] = args[i:]