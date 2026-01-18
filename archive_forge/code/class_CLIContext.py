import json
import numbers
from collections import OrderedDict
from functools import update_wrapper
from pprint import pformat
from typing import Any
import click
from click import Context, ParamType
from kombu.utils.objects import cached_property
from celery._state import get_current_app
from celery.signals import user_preload_options
from celery.utils import text
from celery.utils.log import mlevel
from celery.utils.time import maybe_iso8601
class CLIContext:
    """Context Object for the CLI."""

    def __init__(self, app, no_color, workdir, quiet=False):
        """Initialize the CLI context."""
        self.app = app or get_current_app()
        self.no_color = no_color
        self.quiet = quiet
        self.workdir = workdir

    @cached_property
    def OK(self):
        return self.style('OK', fg='green', bold=True)

    @cached_property
    def ERROR(self):
        return self.style('ERROR', fg='red', bold=True)

    def style(self, message=None, **kwargs):
        if self.no_color:
            return message
        else:
            return click.style(message, **kwargs)

    def secho(self, message=None, **kwargs):
        if self.no_color:
            kwargs['color'] = False
            click.echo(message, **kwargs)
        else:
            click.secho(message, **kwargs)

    def echo(self, message=None, **kwargs):
        if self.no_color:
            kwargs['color'] = False
            click.echo(message, **kwargs)
        else:
            click.echo(message, **kwargs)

    def error(self, message=None, **kwargs):
        kwargs['err'] = True
        if self.no_color:
            kwargs['color'] = False
            click.echo(message, **kwargs)
        else:
            click.secho(message, **kwargs)

    def pretty(self, n):
        if isinstance(n, list):
            return (self.OK, self.pretty_list(n))
        if isinstance(n, dict):
            if 'ok' in n or 'error' in n:
                return self.pretty_dict_ok_error(n)
            else:
                s = json.dumps(n, sort_keys=True, indent=4)
                if not self.no_color:
                    s = highlight(s, LEXER, FORMATTER)
                return (self.OK, s)
        if isinstance(n, str):
            return (self.OK, n)
        return (self.OK, pformat(n))

    def pretty_list(self, n):
        if not n:
            return '- empty -'
        return '\n'.join((f'{self.style('*', fg='white')} {item}' for item in n))

    def pretty_dict_ok_error(self, n):
        try:
            return (self.OK, text.indent(self.pretty(n['ok'])[1], 4))
        except KeyError:
            pass
        return (self.ERROR, text.indent(self.pretty(n['error'])[1], 4))

    def say_chat(self, direction, title, body='', show_body=False):
        if direction == '<-' and self.quiet:
            return
        dirstr = not self.quiet and f'{self.style(direction, fg='white', bold=True)} ' or ''
        self.echo(f'{dirstr} {title}')
        if body and show_body:
            self.echo(body)