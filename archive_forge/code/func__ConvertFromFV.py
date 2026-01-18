from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from utils import bq_error
def _ConvertFromFV(self, schema, row):
    """Converts from FV format to possibly nested lists of values."""
    if not row:
        return None
    values = [entry.get('v', '') for entry in row.get('f', [])]
    result = []
    for field, v in zip(schema, values):
        if 'type' not in field:
            raise bq_error.BigqueryCommunicationError('Invalid response: missing type property')
        if field['type'].upper() == 'RECORD':
            subfields = field.get('fields', [])
            if field.get('mode', 'NULLABLE').upper() == 'REPEATED':
                result.append([self._ConvertFromFV(subfields, subvalue.get('v', '')) for subvalue in v])
            else:
                result.append(self._ConvertFromFV(subfields, v))
        elif field.get('mode', 'NULLABLE').upper() == 'REPEATED':
            result.append([subvalue.get('v', '') for subvalue in v])
        else:
            result.append(v)
    return result