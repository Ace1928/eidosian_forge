from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
class DosInfoExternal(validation.Validated):
    """Describes the format of a dos.yaml file."""
    ATTRIBUTES = {appinfo.APPLICATION: validation.Optional(appinfo.APPLICATION_RE_STRING), BLACKLIST: validation.Optional(validation.Repeated(BlacklistEntry))}