import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers
def _IsMapEntry(field):
    return field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and field.message_type.has_options and field.message_type.GetOptions().map_entry