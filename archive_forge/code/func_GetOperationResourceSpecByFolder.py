from googlecloudsdk.calliope.concepts import concepts
def GetOperationResourceSpecByFolder():
    return concepts.ResourceSpec('auditmanager.folders.locations.operationDetails', resource_name='operation', operationDetailsId=OperationAttributeConfig(), locationsId=LocationAttributeConfig(), foldersId=FolderAttributeConfig())