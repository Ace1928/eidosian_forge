from __future__ import annotations
import typing
from copy import copy
import numpy as np
from ..iapi import panel_ranges

        Backtransform the panel range in panel_params to data coordinates

        Coordinate systems that do any transformations should override
        this method. e.g. coord_trans has to override this method.
        