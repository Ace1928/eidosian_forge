from libcloud.common.types import LibcloudError
class ContainerDoesNotExistError(ContainerError):
    error_type = 'ContainerDoesNotExistError'