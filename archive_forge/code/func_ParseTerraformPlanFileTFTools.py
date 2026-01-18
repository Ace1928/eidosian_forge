import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def ParseTerraformPlanFileTFTools(proposed_changes_file):
    """Parses and converts the tf plan file into CAI Format."""
    env_vars = {'GOOGLE_OAUTH_ACCESS_TOKEN': GetFreshAccessToken(account=properties.VALUES.core.account.Get()), 'USE_STRUCTURED_LOGGING': 'true'}
    operation_result = TerraformToolsBinaryOperation()(command='tfplan-to-cai', terraform_plan_json=proposed_changes_file, project=properties.VALUES.core.project.Get(), env=env_vars)
    if operation_result.stderr:
        handler = binary_operations.DefaultStreamStructuredErrHandler(None)
        for line in operation_result.stderr.split('\n'):
            handler(line)
    return json.loads(operation_result.stdout)