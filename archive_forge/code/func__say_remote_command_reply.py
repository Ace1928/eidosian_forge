from functools import partial
from typing import Literal
import click
from kombu.utils.json import dumps
from celery.bin.base import COMMA_SEPARATED_LIST, CeleryCommand, CeleryOption, handle_preload_options
from celery.exceptions import CeleryCommandException
from celery.platforms import EX_UNAVAILABLE
from celery.utils import text
from celery.worker.control import Panel
def _say_remote_command_reply(ctx, replies, show_reply=False):
    node = next(iter(replies))
    reply = replies[node]
    node = ctx.obj.style(f'{node}: ', fg='cyan', bold=True)
    status, preply = ctx.obj.pretty(reply)
    ctx.obj.say_chat('->', f'{node}{status}', text.indent(preply, 4) if show_reply else '', show_body=show_reply)