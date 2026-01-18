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
class CeleryCommand(click.Command):
    """Customized command for Celery."""

    def format_options(self, ctx, formatter):
        """Write all the options into the formatter if they exist."""
        opts = OrderedDict()
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                if hasattr(param, 'help_group') and param.help_group:
                    opts.setdefault(str(param.help_group), []).append(rv)
                else:
                    opts.setdefault('Options', []).append(rv)
        for name, opts_group in opts.items():
            with formatter.section(name):
                formatter.write_dl(opts_group)