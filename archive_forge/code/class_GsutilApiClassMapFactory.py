from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.boto_translation import BotoTranslation
from gslib.gcs_json_api import GcsJsonApi
class GsutilApiClassMapFactory(object):
    """Factory for generating gsutil API class maps.

  A valid class map is defined as:
    {
      (key) Provider prefix used in URI strings.
      (value) {
        (key) ApiSelector describing the API format.
        (value) CloudApi child class that implements this API.
      }
    }
  """

    @classmethod
    def GetClassMap(cls):
        """Returns the default gsutil class map."""
        gs_class_map = {ApiSelector.XML: BotoTranslation, ApiSelector.JSON: GcsJsonApi}
        s3_class_map = {ApiSelector.XML: BotoTranslation}
        class_map = {'gs': gs_class_map, 's3': s3_class_map}
        return class_map