from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DLP(base.Group):
    """Manage sensitive data with Cloud Data Loss Prevention.

  The DLP API lets you understand and manage sensitive data. It provides
  fast, scalable classification and optional redaction for sensitive data
  elements like credit card numbers, names, Social Security numbers, passport
  numbers, U.S. and selected international driver's license numbers, and phone
  numbers. The API classifies this data using more than 50 predefined detectors
  to identify patterns, formats, and checksums, and even understands contextual
  clues. The API supports text and images; just send data to the API or
  specify data stored on your Google Cloud Storage, BigQuery,
  or Cloud Datastore instances.
  """
    category = base.SECURITY_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args