from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
class ProvJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, ProvDocument):
            return encode_json_document(o)
        else:
            return super(ProvJSONEncoder, self).encode(o)