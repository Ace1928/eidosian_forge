import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
def _definition_factory(self, id):
    """Turn a resource type ID into a ResourceType."""
    return Resource(self.application, self.parameter.get_value(), self.application.resource_types.get(id).tag)