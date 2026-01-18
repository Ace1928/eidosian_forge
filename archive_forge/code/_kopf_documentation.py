from __future__ import annotations
import kopf
import copy
import datetime
import aiohttp
from kopf._cogs.clients import api, errors
from kopf._cogs.configs import configuration
from kopf._cogs.helpers import typedefs
from kopf._cogs.structs import bodies, references
import kopf._cogs.clients.events
from typing import Dict, Any, TYPE_CHECKING
from lazyops.utils.logs import default_logger as _logger

    Issue an event for the object.
    This is where they can also be accumulated, aggregated, grouped,
    and where the rate-limits should be maintained. It can (and should)
    be done by the client library, as it is done in the Go client.
    