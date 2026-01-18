import os
import sys
import click
from click import ParamType
from click.types import StringParamType
from celery import concurrency
from celery.bin.base import (COMMA_SEPARATED_LIST, LOG_LEVEL, CeleryDaemonCommand, CeleryOption,
from celery.concurrency.base import BasePool
from celery.exceptions import SecurityError
from celery.platforms import EX_FAILURE, EX_OK, detached, maybe_drop_privileges
from celery.utils.log import get_logger
from celery.utils.nodenames import default_nodename, host_format, node_format
class Autoscale(ParamType):
    """Autoscaling parameter."""
    name = '<min workers>, <max workers>'

    def convert(self, value, param, ctx):
        value = value.split(',')
        if len(value) > 2:
            self.fail(f'Expected two comma separated integers or one integer.Got {len(value)} instead.')
        if len(value) == 1:
            try:
                value = (int(value[0]), 0)
            except ValueError:
                self.fail(f'Expected an integer. Got {value} instead.')
        try:
            return tuple(reversed(sorted(map(int, value))))
        except ValueError:
            self.fail(f'Expected two comma separated integers.Got {value.join(',')} instead.')