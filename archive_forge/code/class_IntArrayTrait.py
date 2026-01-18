from unittest import TestCase
from traitlets import HasTraits, TraitError, observe, Undefined
from traitlets.tests.test_traitlets import TraitTestBase
from traittypes import Array, DataFrame, Series, Dataset, DataArray
import numpy as np
import pandas as pd
import xarray as xr
class IntArrayTrait(HasTraits):
    value = Array().tag(dtype=np.int)