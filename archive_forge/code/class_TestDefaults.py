import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import (
from sklearn.linear_model import LinearRegression
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.validation import check_is_fitted
class TestDefaults(_MetadataRequester):
    __metadata_request__fit = {'sample_weight': None, 'my_other_param': None}
    __metadata_request__score = {'sample_weight': None, 'my_param': True, 'my_other_param': None}
    __metadata_request__predict = {'my_param': True}