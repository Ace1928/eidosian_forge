import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_template, source='a Ray Dataset', params='ray_obj : ray.data.Dataset', method='modin.core.execution.ray.implementations.pandas_on_ray.io.PandasOnRayIO.from_ray')
def _from_ray(cls, ray_obj):
    return cls.io_cls.from_ray(ray_obj)