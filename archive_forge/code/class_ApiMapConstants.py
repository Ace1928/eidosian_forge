from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.boto_translation import BotoTranslation
from gslib.gcs_json_api import GcsJsonApi
class ApiMapConstants(object):
    """Enum class for API map entries."""
    API_MAP = 'apiclass'
    SUPPORT_MAP = 'supported'
    DEFAULT_MAP = 'default'