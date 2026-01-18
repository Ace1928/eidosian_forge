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
def _method_tag_iter(self):
    """Iterate over this resource's <method> tags."""
    definition = self.resolve_definition().tag
    for method_tag in definition.findall(wadl_xpath('method')):
        yield method_tag