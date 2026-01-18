from copy import deepcopy
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from ..exceptions import NotFittedError
from ..utils._param_validation import HasMethods, Interval, Options
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.validation import _num_features, check_is_fitted, check_scalar
from ._base import SelectorMixin, _get_feature_importances
Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        