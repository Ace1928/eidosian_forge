from libcloud.common.types import LibcloudError
class ContainerAlreadyExistsError(ContainerError):
    error_type = 'ContainerAlreadyExistsError'