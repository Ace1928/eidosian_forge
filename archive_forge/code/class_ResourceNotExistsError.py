import botocore.exceptions
class ResourceNotExistsError(Boto3Error, botocore.exceptions.DataNotFoundError):
    """Raised when you attempt to create a resource that does not exist."""

    def __init__(self, service_name, available_services, has_low_level_client):
        msg = "The '%s' resource does not exist.\nThe available resources are:\n   - %s\n" % (service_name, '\n   - '.join(available_services))
        if has_low_level_client:
            msg += "\nConsider using a boto3.client('%s') instead of a resource for '%s'" % (service_name, service_name)
        Boto3Error.__init__(self, msg)