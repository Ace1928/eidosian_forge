from __future__ import annotations
import json
from typing import (
from pandas.io.excel._base import ExcelWriter
from pandas.io.excel._util import (
class XlsxWriter(ExcelWriter):
    _engine = 'xlsxwriter'
    _supported_extensions = ('.xlsx',)

    def __init__(self, path: FilePath | WriteExcelBuffer | ExcelWriter, engine: str | None=None, date_format: str | None=None, datetime_format: str | None=None, mode: str='w', storage_options: StorageOptions | None=None, if_sheet_exists: ExcelWriterIfSheetExists | None=None, engine_kwargs: dict[str, Any] | None=None, **kwargs) -> None:
        from xlsxwriter import Workbook
        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)
        if mode == 'a':
            raise ValueError('Append mode is not supported with xlsxwriter!')
        super().__init__(path, engine=engine, date_format=date_format, datetime_format=datetime_format, mode=mode, storage_options=storage_options, if_sheet_exists=if_sheet_exists, engine_kwargs=engine_kwargs)
        try:
            self._book = Workbook(self._handles.handle, **engine_kwargs)
        except TypeError:
            self._handles.handle.close()
            raise

    @property
    def book(self):
        """
        Book instance of class xlsxwriter.Workbook.

        This attribute can be used to access engine-specific features.
        """
        return self._book

    @property
    def sheets(self) -> dict[str, Any]:
        result = self.book.sheetnames
        return result

    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        self.book.close()

    def _write_cells(self, cells, sheet_name: str | None=None, startrow: int=0, startcol: int=0, freeze_panes: tuple[int, int] | None=None) -> None:
        sheet_name = self._get_sheet_name(sheet_name)
        wks = self.book.get_worksheet_by_name(sheet_name)
        if wks is None:
            wks = self.book.add_worksheet(sheet_name)
        style_dict = {'null': None}
        if validate_freeze_panes(freeze_panes):
            wks.freeze_panes(*freeze_panes)
        for cell in cells:
            val, fmt = self._value_with_fmt(cell.val)
            stylekey = json.dumps(cell.style)
            if fmt:
                stylekey += fmt
            if stylekey in style_dict:
                style = style_dict[stylekey]
            else:
                style = self.book.add_format(_XlsxStyler.convert(cell.style, fmt))
                style_dict[stylekey] = style
            if cell.mergestart is not None and cell.mergeend is not None:
                wks.merge_range(startrow + cell.row, startcol + cell.col, startrow + cell.mergestart, startcol + cell.mergeend, val, style)
            else:
                wks.write(startrow + cell.row, startcol + cell.col, val, style)