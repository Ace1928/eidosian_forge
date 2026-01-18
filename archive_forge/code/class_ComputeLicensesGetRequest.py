from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeLicensesGetRequest(_messages.Message):
    """A ComputeLicensesGetRequest object.

  Fields:
    license: Name of the License resource to return.
    project: Project ID for this request.
  """
    license = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)