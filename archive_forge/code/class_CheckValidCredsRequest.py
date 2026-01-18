from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckValidCredsRequest(_messages.Message):
    """A request to determine whether the user has valid credentials. This
  method is used to limit the number of OAuth popups in the user interface.
  The user id is inferred from the API call context. If the data source has
  the Google+ authorization type, this method returns false, as it cannot be
  determined whether the credentials are already valid merely based on the
  user id.
  """