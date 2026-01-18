import warnings
from io import BytesIO
import pandas
from pandas.util._decorators import doc
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.parsers import (
@doc(_doc_pandas_parser_class, data_type='custom text')
class ExperimentalCustomTextParser(PandasParser):

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        return PandasParser.generic_parse(fname, **kwargs)