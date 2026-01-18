import csv
import datetime
from decimal import Decimal
import json
import operator
import sys
import uuid
from peewee import *
from playhouse.db_url import connect
from playhouse.migrate import migrate
from playhouse.migrate import SchemaMigrator
from playhouse.reflection import Introspector
class JSONExporter(Exporter):

    def __init__(self, query, iso8601_datetimes=False):
        super(JSONExporter, self).__init__(query)
        self.iso8601_datetimes = iso8601_datetimes

    def _make_default(self):
        datetime_types = (datetime.datetime, datetime.date, datetime.time)
        if self.iso8601_datetimes:

            def default(o):
                if isinstance(o, datetime_types):
                    return o.isoformat()
                elif isinstance(o, (Decimal, uuid.UUID)):
                    return str(o)
                raise TypeError('Unable to serialize %r as JSON' % o)
        else:

            def default(o):
                if isinstance(o, datetime_types + (Decimal, uuid.UUID)):
                    return str(o)
                raise TypeError('Unable to serialize %r as JSON' % o)
        return default

    def export(self, file_obj, **kwargs):
        json.dump(list(self.query), file_obj, default=self._make_default(), **kwargs)