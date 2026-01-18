from libcloud.common.types import LibcloudError
class ZoneAlreadyExistsError(ZoneError):
    error_type = 'ZoneAlreadyExistsError'