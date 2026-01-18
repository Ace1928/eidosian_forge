from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
class EntityType(enum.Enum):
    """Cloud Entity Types."""
    ORGANIZATION = 1
    FOLDER = 2
    PROJECT = 3
    BILLING_ACCOUNT = 4