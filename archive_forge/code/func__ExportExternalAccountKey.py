from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.publicca import base as publicca_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def _ExportExternalAccountKey(external_account_key, key_output_file):
    try:
        files.WriteFileContents(key_output_file, external_account_key)
    except (files.Error, OSError, IOError):
        raise exceptions.BadFileException("Could not write external account key to '{}'.".format(key_output_file))