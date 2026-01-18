import datetime
from decimal import Decimal
import json
from sys import stdout
from uuid import UUID
import fastavro as avro
class CleanJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif isinstance(obj, (Decimal, UUID)):
            return str(obj)
        elif isinstance(obj, bytes):
            return obj.decode('iso-8859-1')
        else:
            return json.JSONEncoder.default(self, obj)