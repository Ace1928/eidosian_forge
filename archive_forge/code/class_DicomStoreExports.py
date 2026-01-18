from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DicomStoreExports(base.Group):
    """Manage Cloud Healthcare API DICOM store exports.

  Commands for exporting Google Cloud Healthcare API DICOM stores.
  """