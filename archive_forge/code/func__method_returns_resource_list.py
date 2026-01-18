from botocore.docs.method import document_model_driven_method
def _method_returns_resource_list(resource):
    for identifier in resource.identifiers:
        if identifier.path and '[]' in identifier.path:
            return True
    return False