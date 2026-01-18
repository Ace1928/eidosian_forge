import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
class LogCommandGroup(click.Group):

    def resolve_command(self, ctx, args):
        """Try resolve the command line args assuming users omitted the subcommand.

        This overrides the default `resolve_command` for the parent class.
        This will allow command alias of `ray <glob>` to `ray cluster <glob>`.
        """
        ctx.resilient_parsing = True
        res = super().resolve_command(ctx, args)
        cmd_name, cmd, parsed_args = res
        if cmd is None:
            return super().resolve_command(ctx, ['cluster'] + args)
        return (cmd_name, cmd, parsed_args)