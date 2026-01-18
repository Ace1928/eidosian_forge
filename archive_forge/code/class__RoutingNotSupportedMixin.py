import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
class _RoutingNotSupportedMixin:
    """A mixin to be used to remove the default `get_metadata_routing`.

    This is used in meta-estimators where metadata routing is not yet
    implemented.

    This also makes it clear in our rendered documentation that this method
    cannot be used.
    """

    def get_metadata_routing(self):
        """Raise `NotImplementedError`.

        This estimator does not support metadata routing yet."""
        raise NotImplementedError(f'{self.__class__.__name__} has not implemented metadata routing yet.')