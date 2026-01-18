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
def _IsWrapperMessage(message_descriptor):
    return message_descriptor.file.name == 'google/protobuf/wrappers.proto'