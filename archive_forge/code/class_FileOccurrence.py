from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileOccurrence(_messages.Message):
    """FileOccurrence represents an SPDX File Information section:
  https://spdx.github.io/spdx-spec/4-file-information/

  Fields:
    attributions: This field provides a place for the SPDX data creator to
      record, at the file level, acknowledgements that may be needed to be
      communicated in some contexts
    comment: This field provides a place for the SPDX file creator to record
      any general comments about the file
    contributors: This field provides a place for the SPDX file creator to
      record file contributors
    copyright: Identify the copyright holder of the file, as well as any dates
      present
    filesLicenseInfo: This field contains the license information actually
      found in the file, if any
    id: Uniquely identify any element in an SPDX document which may be
      referenced by other elements
    licenseConcluded: This field contains the license the SPDX file creator
      has concluded as governing the file or alternative values if the
      governing license cannot be determined
    notice: This field provides a place for the SPDX file creator to record
      license notices or other such related notices found in the file
  """
    attributions = _messages.StringField(1, repeated=True)
    comment = _messages.StringField(2)
    contributors = _messages.StringField(3, repeated=True)
    copyright = _messages.StringField(4)
    filesLicenseInfo = _messages.StringField(5, repeated=True)
    id = _messages.StringField(6)
    licenseConcluded = _messages.MessageField('License', 7)
    notice = _messages.StringField(8)