import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def _routing_enabled():
    """Return whether metadata routing is enabled.

    .. versionadded:: 1.3

    Returns
    -------
    enabled : bool
        Whether metadata routing is enabled. If the config is not set, it
        defaults to False.
    """
    return get_config().get('enable_metadata_routing', False)