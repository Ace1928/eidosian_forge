import warnings
from io import BytesIO
import pandas
from pandas.util._decorators import doc
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.parsers import (
@doc(_doc_pandas_parser_class, data_type='multiple CSV files simultaneously')
class ExperimentalPandasCSVGlobParser(PandasCSVParser):

    @staticmethod
    @doc(_doc_parse_func, parameters='chunks : list\n    List, where each element of the list is a list of tuples. The inner lists\n    of tuples contains the data file name of the chunk, chunk start offset, and\n    chunk end offsets for its corresponding file.')
    def parse(chunks, **kwargs):
        warnings.filterwarnings('ignore')
        num_splits = kwargs.pop('num_splits', None)
        index_col = kwargs.get('index_col', None)
        if isinstance(chunks, str):
            return pandas.read_csv(chunks, **kwargs)
        compression = kwargs.pop('compression', 'infer')
        storage_options = kwargs.pop('storage_options', None) or {}
        pandas_dfs = []
        for fname, start, end in chunks:
            if start is not None and end is not None:
                with OpenFile(fname, 'rb', compression, **storage_options) as bio:
                    if kwargs.get('encoding', None) is not None:
                        header = b'' + bio.readline()
                    else:
                        header = b''
                    bio.seek(start)
                    to_read = header + bio.read(end - start)
                pandas_dfs.append(pandas.read_csv(BytesIO(to_read), **kwargs))
            else:
                return pandas.read_csv(fname, compression=compression, storage_options=storage_options, **kwargs)
        if len(pandas_dfs) > 1:
            pandas_df = pandas.concat(pandas_dfs)
        elif len(pandas_dfs) > 0:
            pandas_df = pandas_dfs[0]
        else:
            pandas_df = pandas.DataFrame()
        if index_col is not None:
            index = pandas_df.index
        else:
            index = len(pandas_df)
        return _split_result_for_readers(1, num_splits, pandas_df) + [index, pandas_df.dtypes]