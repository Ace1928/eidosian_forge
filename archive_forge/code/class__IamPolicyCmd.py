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
class _IamPolicyCmd(bigquery_command.BigqueryCmd):
    """Common superclass for commands that interact with BQ's IAM meta API.

  Provides common flags, identifier decoding logic, and GetIamPolicy and
  SetIamPolicy logic.
  """

    def __init__(self, name: str, fv: flags.FlagValues, verb):
        """Initialize.

    Args:
      name: the command name string to bind to this handler class.
      fv: the FlagValues flag-registry object.
      verb: the verb string (e.g. 'Set', 'Get', 'Add binding to', ...) to print
        in various descriptions.
    """
        super(_IamPolicyCmd, self).__init__(name, fv)
        self.surface_in_shell = False
        flags.DEFINE_boolean('dataset', False, '%s IAM policy for dataset described by this identifier.' % verb, short_name='d', flag_values=fv)
        flags.DEFINE_boolean('table', False, '%s IAM policy for table described by this identifier.' % verb, short_name='t', flag_values=fv)
        flags.DEFINE_boolean('connection', None, '%s IAM policy for connection described by this identifier.' % verb, flag_values=fv)

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

    def GetPolicyForReference(self, client, reference):
        """Get the IAM policy for a table or dataset.

    Args:
      reference: A DatasetReference or TableReference.

    Returns:
      The policy object, composed of dictionaries, lists, and primitive types.

    Raises:
      RuntimeError: reference isn't an expected type.
    """
        if isinstance(reference, bq_id_utils.ApiClientHelper.TableReference):
            return client.GetTableIAMPolicy(reference)
        elif isinstance(reference, bq_id_utils.ApiClientHelper.DatasetReference):
            return client.GetDatasetIAMPolicy(reference)
        elif isinstance(reference, bq_id_utils.ApiClientHelper.ConnectionReference):
            return client.GetConnectionIAMPolicy(reference)
        raise RuntimeError('Unexpected reference type: {r_type}'.format(r_type=type(reference)))

    def SetPolicyForReference(self, client, reference, policy):
        """Set the IAM policy for a table or dataset.

    Args:
      reference: A DatasetReference or TableReference.
      policy: The policy object, composed of dictionaries, lists, and primitive
        types.

    Raises:
      RuntimeError: reference isn't an expected type.
    """
        if isinstance(reference, bq_id_utils.ApiClientHelper.TableReference):
            return client.SetTableIAMPolicy(reference, policy)
        elif isinstance(reference, bq_id_utils.ApiClientHelper.DatasetReference):
            return client.SetDatasetIAMPolicy(reference, policy)
        elif isinstance(reference, bq_id_utils.ApiClientHelper.ConnectionReference):
            return client.SetConnectionIAMPolicy(reference, policy)
        raise RuntimeError('Unexpected reference type: {r_type}'.format(r_type=type(reference)))