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
@summary_state_cli_group.command(name='actors')
@timeout_option
@address_option
@click.pass_context
@PublicAPI(stability='stable')
def actor_summary(ctx, timeout: float, address: str):
    """Summarize the actor state of the cluster.

    By default, the output contains the information grouped by
    actor class names.

    The output schema is
    :class:`ray.util.state.common.ActorSummaries
    <ray.util.state.common.ActorSummaries>`.

    Raises:
        :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>`
            if the CLI is failed to query the data.
    """
    print(format_summary_output(summarize_actors(address=address, timeout=timeout, raise_on_missing_output=False, _explain=True), resource=StateResource.ACTORS))