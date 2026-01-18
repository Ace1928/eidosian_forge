from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from typing import Optional
from absl import app
from absl import flags
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from utils import bq_id_utils
def GetReferenceFromIdentifier(self, client, identifier):
    provided_flags = [f for f in [self.d, self.t, self.connection] if f is not None and f]
    if len(provided_flags) > 1:
        raise app.UsageError('Cannot specify more than one of -d, -t or -connection.')
    if not identifier:
        raise app.UsageError('Must provide an identifier for %s.' % (self._command_name,))
    if self.t:
        reference = client.GetTableReference(identifier)
    elif self.d:
        reference = client.GetDatasetReference(identifier)
    elif self.connection:
        reference = client.GetConnectionReference(identifier, default_location=FLAGS.location)
    else:
        reference = client.GetReference(identifier)
        bq_id_utils.typecheck(reference, (bq_id_utils.ApiClientHelper.DatasetReference, bq_id_utils.ApiClientHelper.TableReference), 'Invalid identifier "%s" for %s.' % (identifier, self._command_name), is_usage_error=True)
    return reference