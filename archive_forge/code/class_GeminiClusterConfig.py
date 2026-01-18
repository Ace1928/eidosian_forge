from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GeminiClusterConfig(_messages.Message):
    """Cluster level configuration parameters related to the Gemini in
  Databases add-on. See go/prd-enable-duet-ai-databases for more details.

  Fields:
    entitled: Output only. Whether the Gemini in Databases add-on is enabled
      for the cluster. It will be true only if the add-on has been enabled for
      the billing account corresponding to the cluster. Its status is toggled
      from the Admin Control Center (ACC) and cannot be toggled using
      AlloyDB's APIs.
  """
    entitled = _messages.BooleanField(1)