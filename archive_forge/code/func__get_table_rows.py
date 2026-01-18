from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def _get_table_rows(self, raw_output):
    table = self.parser.table(raw_output)
    rows = []
    for row in table['values']:
        rows.append(row[0])
    return rows