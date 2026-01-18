from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
from apitools.base.py import extra_types
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_printer
def _FormatRowTable(self, resp):
    """Formats rows in a [QueryAssetsResponse]'s queryResults into a table.

    Args:
      resp: The [QueryAssetsResponse] that contains 0 or more rows.

    Returns:
      A 'Lines' custom printer object that corresponds to the formatted table
      when printed out.

    The response.queryResult.rows in response:
    {
      "jobReference":
      "CiBqb2JfdDR2SFgwa3BPNFpQVDFudVJJaW5TdVNfS1N0YxIBAxjH8ZmAo6qckik",
      "done": true,
      "queryResult": {
        "rows": [
          {
            "f": [
              {
                "v":
                "//cloudresourcemanager.googleapis.com/folders/417243649856"
              }
            ]
          }
        ],
        "schema": {
          "fields": [
            {
              "field": "name",
              "type": "STRING",
              "mode": "NULLABLE"
            }
          ]
        },
        "total_rows": 1
      }
    }
    Will return a custom printer Lines object holding the following string:
    ┌────────────────────────────────────────────────────────────┐
    │                            name                            │
    ├────────────────────────────────────────────────────────────┤
    │ //cloudresourcemanager.googleapis.com/folders/417243649856 │
    └────────────────────────────────────────────────────────────┘
    """
    if not hasattr(resp, 'queryResult') or not hasattr(resp.queryResult, 'schema'):
        return None
    schema = resp.queryResult.schema
    rows = resp.queryResult.rows
    row_list = []
    if not schema.fields:
        return None
    schemabuf = io.StringIO()
    schemabuf.write('table[box]({})'.format(', '.join(('{}:label={}'.format(field.field, field.field) for field in schema.fields))))
    for row in rows:
        row_json = extra_types.encoding.MessageToPyValue(row)
        schema_json = extra_types.encoding.MessageToPyValue(schema)
        row_list.append(self._ConvertFromFV(schema_json, row_json, False))
    raw_out = io.StringIO()
    resource_printer.Print(row_list, schemabuf.getvalue(), out=raw_out)
    return cp.Lines([raw_out.getvalue()])