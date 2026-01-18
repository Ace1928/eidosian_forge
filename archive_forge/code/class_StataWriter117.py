from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
class StataWriter117(StataWriter):
    """
    A class for writing Stata binary dta files in Stata 13 format (117)

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter117 instance
        The StataWriter117 instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1, 'a']], columns=['a', 'b', 'c'])
    >>> writer = pd.io.stata.StataWriter117('./data_file.dta', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = pd.io.stata.StataWriter117(
    ...     './data_file.zip', data, compression=compression
    ...     )
    >>> writer.write_file()

    Or with long strings stored in strl format
    >>> data = pd.DataFrame([['A relatively long string'], [''], ['']],
    ...                     columns=['strls'])
    >>> writer = pd.io.stata.StataWriter117(
    ...     './data_file_with_long_strings.dta', data, convert_strl=['strls'])
    >>> writer.write_file()
    """
    _max_string_length = 2045
    _dta_version = 117

    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None=None, write_index: bool=True, byteorder: str | None=None, time_stamp: datetime | None=None, data_label: str | None=None, variable_labels: dict[Hashable, str] | None=None, convert_strl: Sequence[Hashable] | None=None, compression: CompressionOptions='infer', storage_options: StorageOptions | None=None, *, value_labels: dict[Hashable, dict[float, str]] | None=None) -> None:
        self._convert_strl: list[Hashable] = []
        if convert_strl is not None:
            self._convert_strl.extend(convert_strl)
        super().__init__(fname, data, convert_dates, write_index, byteorder=byteorder, time_stamp=time_stamp, data_label=data_label, variable_labels=variable_labels, value_labels=value_labels, compression=compression, storage_options=storage_options)
        self._map: dict[str, int] = {}
        self._strl_blob = b''

    @staticmethod
    def _tag(val: str | bytes, tag: str) -> bytes:
        """Surround val with <tag></tag>"""
        if isinstance(val, str):
            val = bytes(val, 'utf-8')
        return bytes('<' + tag + '>', 'utf-8') + val + bytes('</' + tag + '>', 'utf-8')

    def _update_map(self, tag: str) -> None:
        """Update map location for tag with file position"""
        assert self.handles.handle is not None
        self._map[tag] = self.handles.handle.tell()

    def _write_header(self, data_label: str | None=None, time_stamp: datetime | None=None) -> None:
        """Write the file header"""
        byteorder = self._byteorder
        self._write_bytes(bytes('<stata_dta>', 'utf-8'))
        bio = BytesIO()
        bio.write(self._tag(bytes(str(self._dta_version), 'utf-8'), 'release'))
        bio.write(self._tag(byteorder == '>' and 'MSF' or 'LSF', 'byteorder'))
        nvar_type = 'H' if self._dta_version <= 118 else 'I'
        bio.write(self._tag(struct.pack(byteorder + nvar_type, self.nvar), 'K'))
        nobs_size = 'I' if self._dta_version == 117 else 'Q'
        bio.write(self._tag(struct.pack(byteorder + nobs_size, self.nobs), 'N'))
        label = data_label[:80] if data_label is not None else ''
        encoded_label = label.encode(self._encoding)
        label_size = 'B' if self._dta_version == 117 else 'H'
        label_len = struct.pack(byteorder + label_size, len(encoded_label))
        encoded_label = label_len + encoded_label
        bio.write(self._tag(encoded_label, 'label'))
        if time_stamp is None:
            time_stamp = datetime.now()
        elif not isinstance(time_stamp, datetime):
            raise ValueError('time_stamp should be datetime type')
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_lookup = {i + 1: month for i, month in enumerate(months)}
        ts = time_stamp.strftime('%d ') + month_lookup[time_stamp.month] + time_stamp.strftime(' %Y %H:%M')
        stata_ts = b'\x11' + bytes(ts, 'utf-8')
        bio.write(self._tag(stata_ts, 'timestamp'))
        self._write_bytes(self._tag(bio.getvalue(), 'header'))

    def _write_map(self) -> None:
        """
        Called twice during file write. The first populates the values in
        the map with 0s.  The second call writes the final map locations when
        all blocks have been written.
        """
        if not self._map:
            self._map = {'stata_data': 0, 'map': self.handles.handle.tell(), 'variable_types': 0, 'varnames': 0, 'sortlist': 0, 'formats': 0, 'value_label_names': 0, 'variable_labels': 0, 'characteristics': 0, 'data': 0, 'strls': 0, 'value_labels': 0, 'stata_data_close': 0, 'end-of-file': 0}
        self.handles.handle.seek(self._map['map'])
        bio = BytesIO()
        for val in self._map.values():
            bio.write(struct.pack(self._byteorder + 'Q', val))
        self._write_bytes(self._tag(bio.getvalue(), 'map'))

    def _write_variable_types(self) -> None:
        self._update_map('variable_types')
        bio = BytesIO()
        for typ in self.typlist:
            bio.write(struct.pack(self._byteorder + 'H', typ))
        self._write_bytes(self._tag(bio.getvalue(), 'variable_types'))

    def _write_varnames(self) -> None:
        self._update_map('varnames')
        bio = BytesIO()
        vn_len = 32 if self._dta_version == 117 else 128
        for name in self.varlist:
            name = self._null_terminate_str(name)
            name = _pad_bytes_new(name[:32].encode(self._encoding), vn_len + 1)
            bio.write(name)
        self._write_bytes(self._tag(bio.getvalue(), 'varnames'))

    def _write_sortlist(self) -> None:
        self._update_map('sortlist')
        sort_size = 2 if self._dta_version < 119 else 4
        self._write_bytes(self._tag(b'\x00' * sort_size * (self.nvar + 1), 'sortlist'))

    def _write_formats(self) -> None:
        self._update_map('formats')
        bio = BytesIO()
        fmt_len = 49 if self._dta_version == 117 else 57
        for fmt in self.fmtlist:
            bio.write(_pad_bytes_new(fmt.encode(self._encoding), fmt_len))
        self._write_bytes(self._tag(bio.getvalue(), 'formats'))

    def _write_value_label_names(self) -> None:
        self._update_map('value_label_names')
        bio = BytesIO()
        vl_len = 32 if self._dta_version == 117 else 128
        for i in range(self.nvar):
            name = ''
            if self._has_value_labels[i]:
                name = self.varlist[i]
            name = self._null_terminate_str(name)
            encoded_name = _pad_bytes_new(name[:32].encode(self._encoding), vl_len + 1)
            bio.write(encoded_name)
        self._write_bytes(self._tag(bio.getvalue(), 'value_label_names'))

    def _write_variable_labels(self) -> None:
        self._update_map('variable_labels')
        bio = BytesIO()
        vl_len = 80 if self._dta_version == 117 else 320
        blank = _pad_bytes_new('', vl_len + 1)
        if self._variable_labels is None:
            for _ in range(self.nvar):
                bio.write(blank)
            self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))
            return
        for col in self.data:
            if col in self._variable_labels:
                label = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError('Variable labels must be 80 characters or fewer')
                try:
                    encoded = label.encode(self._encoding)
                except UnicodeEncodeError as err:
                    raise ValueError(f'Variable labels must contain only characters that can be encoded in {self._encoding}') from err
                bio.write(_pad_bytes_new(encoded, vl_len + 1))
            else:
                bio.write(blank)
        self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))

    def _write_characteristics(self) -> None:
        self._update_map('characteristics')
        self._write_bytes(self._tag(b'', 'characteristics'))

    def _write_data(self, records) -> None:
        self._update_map('data')
        self._write_bytes(b'<data>')
        self._write_bytes(records.tobytes())
        self._write_bytes(b'</data>')

    def _write_strls(self) -> None:
        self._update_map('strls')
        self._write_bytes(self._tag(self._strl_blob, 'strls'))

    def _write_expansion_fields(self) -> None:
        """No-op in dta 117+"""

    def _write_value_labels(self) -> None:
        self._update_map('value_labels')
        bio = BytesIO()
        for vl in self._value_labels:
            lab = vl.generate_value_label(self._byteorder)
            lab = self._tag(lab, 'lbl')
            bio.write(lab)
        self._write_bytes(self._tag(bio.getvalue(), 'value_labels'))

    def _write_file_close_tag(self) -> None:
        self._update_map('stata_data_close')
        self._write_bytes(bytes('</stata_dta>', 'utf-8'))
        self._update_map('end-of-file')

    def _update_strl_names(self) -> None:
        """
        Update column names for conversion to strl if they might have been
        changed to comply with Stata naming rules
        """
        for orig, new in self._converted_names.items():
            if orig in self._convert_strl:
                idx = self._convert_strl.index(orig)
                self._convert_strl[idx] = new

    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """
        Convert columns to StrLs if either very large or in the
        convert_strl variable
        """
        convert_cols = [col for i, col in enumerate(data) if self.typlist[i] == 32768 or col in self._convert_strl]
        if convert_cols:
            ssw = StataStrLWriter(data, convert_cols, version=self._dta_version)
            tab, new_data = ssw.generate_table()
            data = new_data
            self._strl_blob = ssw.generate_blob(tab)
        return data

    def _set_formats_and_types(self, dtypes: Series) -> None:
        self.typlist = []
        self.fmtlist = []
        for col, dtype in dtypes.items():
            force_strl = col in self._convert_strl
            fmt = _dtype_to_default_stata_fmt(dtype, self.data[col], dta_version=self._dta_version, force_strl=force_strl)
            self.fmtlist.append(fmt)
            self.typlist.append(_dtype_to_stata_type_117(dtype, self.data[col], force_strl))