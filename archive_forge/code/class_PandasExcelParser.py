import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple
import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
from modin.config import MinPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
@doc(_doc_pandas_parser_class, data_type='excel files')
class PandasExcelParser(PandasParser):

    @classmethod
    def get_sheet_data(cls, sheet, convert_float):
        """
        Get raw data from the excel sheet.

        Parameters
        ----------
        sheet : openpyxl.worksheet.worksheet.Worksheet
            Sheet to get data from.
        convert_float : bool
            Whether to convert floats to ints or not.

        Returns
        -------
        list
            List with sheet data.
        """
        return [[cls._convert_cell(cell, convert_float) for cell in row] for row in sheet.rows]

    @classmethod
    def _convert_cell(cls, cell, convert_float):
        """
        Convert excel cell to value.

        Parameters
        ----------
        cell : openpyxl.cell.cell.Cell
            Excel cell to convert.
        convert_float : bool
            Whether to convert floats to ints or not.

        Returns
        -------
        list
            Value that was converted from the excel cell.
        """
        if cell.is_date:
            return cell.value
        elif cell.data_type == 'e':
            return np.nan
        elif cell.data_type == 'b':
            return bool(cell.value)
        elif cell.value is None:
            return ''
        elif cell.data_type == 'n':
            if convert_float:
                val = int(cell.value)
                if val == cell.value:
                    return val
            else:
                return float(cell.value)
        return cell.value

    @staticmethod
    def need_rich_text_param():
        """
        Determine whether a required `rich_text` parameter should be specified for the ``WorksheetReader`` constructor.

        Returns
        -------
        bool
        """
        import openpyxl
        from packaging import version
        return version.parse(openpyxl.__version__) >= version.parse('3.1.0')

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        num_splits = kwargs.pop('num_splits', None)
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        excel_header = kwargs.get('_header')
        sheet_name = kwargs.get('sheet_name', 0)
        footer = b'</sheetData></worksheet>'
        if start is None or end is None:
            return pandas.read_excel(fname, **kwargs)
        _skiprows = kwargs.pop('skiprows')
        import re
        from zipfile import ZipFile
        import openpyxl
        from openpyxl.reader.excel import ExcelReader
        from openpyxl.worksheet._reader import WorksheetReader
        from openpyxl.worksheet.worksheet import Worksheet
        from pandas.core.dtypes.common import is_list_like
        from pandas.io.excel._util import fill_mi_header, maybe_convert_usecols
        from pandas.io.parsers import TextParser
        wb = openpyxl.load_workbook(filename=fname, read_only=True)
        ex = ExcelReader(fname, read_only=True)
        ex.read_manifest()
        ex.read_strings()
        if sheet_name == 0:
            sheet_name = wb.sheetnames[sheet_name]
        ws = Worksheet(wb)
        with ZipFile(fname) as z:
            with z.open('xl/worksheets/{}.xml'.format(sheet_name)) as file:
                file.seek(start)
                bytes_data = file.read(end - start)

        def update_row_nums(match):
            """
            Update the row numbers to start at 1.

            Parameters
            ----------
            match : re.Match object
                The match from the origin `re.sub` looking for row number tags.

            Returns
            -------
            str
                The updated string with new row numbers.

            Notes
            -----
            This is needed because the parser we are using does not scale well if
            the row numbers remain because empty rows are inserted for all "missing"
            rows.
            """
            b = match.group(0)
            return re.sub(b'\\d+', lambda c: str(int(c.group(0).decode('utf-8')) - _skiprows).encode('utf-8'), b)
        bytes_data = re.sub(b'r="[A-Z]*\\d+"', update_row_nums, bytes_data)
        bytesio = BytesIO(excel_header + bytes_data + footer)
        common_args = (ws, bytesio, ex.shared_strings, False)
        if PandasExcelParser.need_rich_text_param():
            reader = WorksheetReader(*common_args, rich_text=False)
        else:
            reader = WorksheetReader(*common_args)
        reader.bind_cells()
        data = PandasExcelParser.get_sheet_data(ws, kwargs.pop('convert_float', True))
        usecols = maybe_convert_usecols(kwargs.pop('usecols', None))
        header = kwargs.pop('header', 0)
        index_col = kwargs.pop('index_col', None)
        skiprows = None
        if is_list_like(header) and len(header) == 1:
            header = header[0]
        if header is not None and is_list_like(header):
            control_row = [True] * len(data[0])
            for row in header:
                data[row], control_row = fill_mi_header(data[row], control_row)
        if is_list_like(index_col):
            if not is_list_like(header):
                offset = 1 + header
            else:
                offset = 1 + max(header)
            if offset < len(data):
                for col in index_col:
                    last = data[offset][col]
                    for row in range(offset + 1, len(data)):
                        if data[row][col] == '' or data[row][col] is None:
                            data[row][col] = last
                        else:
                            last = data[row][col]
        parser = TextParser(data, header=header, index_col=index_col, has_index_names=is_list_like(header) and len(header) > 1, skiprows=skiprows, usecols=usecols, skip_blank_lines=False, **kwargs)
        pandas_df = parser.read()
        if len(pandas_df) > 1 and len(pandas_df.columns) != 0 and pandas_df.isnull().all().all():
            pandas_df = pandas.DataFrame(columns=pandas_df.columns)
        if isinstance(pandas_df.index, pandas.RangeIndex):
            pandas_df.index = pandas.RangeIndex(start=_skiprows, stop=len(pandas_df.index) + _skiprows)
        if index_col is not None:
            index = pandas_df.index
        else:
            index = len(pandas_df)
        return _split_result_for_readers(1, num_splits, pandas_df) + [index, pandas_df.dtypes]