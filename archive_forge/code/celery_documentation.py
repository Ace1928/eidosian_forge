import os
import pathlib
import sys
import traceback
import click
import click.exceptions
from click.types import ParamType
from click_didyoumean import DYMGroup
from click_plugins import with_plugins
from celery import VERSION_BANNER
from celery.app.utils import find_app
from celery.bin.amqp import amqp
from celery.bin.base import CeleryCommand, CeleryOption, CLIContext
from celery.bin.beat import beat
from celery.bin.call import call
from celery.bin.control import control, inspect, status
from celery.bin.events import events
from celery.bin.graph import graph
from celery.bin.list import list_
from celery.bin.logtool import logtool
from celery.bin.migrate import migrate
from celery.bin.multi import multi
from celery.bin.purge import purge
from celery.bin.result import result
from celery.bin.shell import shell
from celery.bin.upgrade import upgrade
from celery.bin.worker import worker
Start celery umbrella command.

    This function is the main entrypoint for the CLI.

    :return: The exit code of the CLI.
    