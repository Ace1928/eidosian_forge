from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import sys
from googlecloudsdk.core.resource import resource_property
class PackageState(*_StrEnum):
    INSTALLED = 'installed'
    REMOVED = 'removed'