from peewee import *
from playhouse.sqlite_ext import JSONField
def _build_column_array(self, model, use_old, use_new, skip_fields=None):
    col_array = []
    for field in model._meta.sorted_fields:
        if field.primary_key:
            continue
        if skip_fields is not None and field.name in skip_fields:
            continue
        column = field.column_name
        new = 'NULL' if not use_new else 'NEW."%s"' % column
        old = 'NULL' if not use_old else 'OLD."%s"' % column
        if isinstance(field, JSONField):
            if use_old:
                old = 'json(%s)' % old
            if use_new:
                new = 'json(%s)' % new
        col_array.append("json_array('%s', %s, %s)" % (column, old, new))
    return ', '.join(col_array)