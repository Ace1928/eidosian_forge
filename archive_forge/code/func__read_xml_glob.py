import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_raw_template, source='XML files', params=_doc_io_method_kwargs_params)
def _read_xml_glob(cls, **kwargs):
    current_execution = get_current_execution()
    if current_execution not in supported_executions:
        raise NotImplementedError(f'`_read_xml_glob()` is not implemented for {current_execution} execution.')
    return cls.io_cls.read_xml_glob(**kwargs)