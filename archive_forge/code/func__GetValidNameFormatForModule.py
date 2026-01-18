from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def _GetValidNameFormatForModule(module_type: constants.CustomModuleType) -> str:
    """Returns a list of name format strings for the given module_type."""
    collections = [f'securitycentermanagement.organizations.locations.{module_type}', f'securitycentermanagement.projects.locations.{module_type}', f'securitycentermanagement.folders.locations.{module_type}']
    return [resources.REGISTRY.GetCollectionInfo(collection).GetPath('') for collection in collections]