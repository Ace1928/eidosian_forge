from __future__ import absolute_import, print_function, division
from petl.util.base import Table, iterdata
from petl.compat import text_type
from petl.errors import ArgumentError as PetlArgError
class GoogleSheetView(Table):
    """Conects to a worksheet and iterates over its rows."""

    def __init__(self, credentials_or_client, spreadsheet, worksheet, cell_range, open_by_key):
        self.auth_info = credentials_or_client
        self.spreadsheet = spreadsheet
        self.worksheet = worksheet
        self.cell_range = cell_range
        self.open_by_key = open_by_key

    def __iter__(self):
        gspread_client = _get_gspread_client(self.auth_info)
        wb = _open_spreadsheet(gspread_client, self.spreadsheet, self.open_by_key)
        ws = _select_worksheet(wb, self.worksheet)
        if self.cell_range is not None:
            return self._yield_by_range(ws)
        return self._yield_all_rows(ws)

    def _yield_all_rows(self, ws):
        for row in ws.get_all_values():
            yield tuple(row)

    def _yield_by_range(self, ws):
        found = ws.get_values(self.cell_range)
        for row in found:
            yield tuple(row)