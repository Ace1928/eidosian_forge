from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentNote(_messages.Message):
    """DocumentNote represents an SPDX Document Creation Information section:
  https://spdx.github.io/spdx-spec/2-document-creation-information/

  Fields:
    dataLicence: Compliance with the SPDX specification includes populating
      the SPDX fields therein with data related to such fields ("SPDX-
      Metadata")
    spdxVersion: Provide a reference number that can be used to understand how
      to parse and interpret the rest of the file
  """
    dataLicence = _messages.StringField(1)
    spdxVersion = _messages.StringField(2)