import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
def GetParentFromResourceName(resource_name):
    list_organization_components = resource_name.split('/')
    return list_organization_components[0] + '/' + list_organization_components[1]