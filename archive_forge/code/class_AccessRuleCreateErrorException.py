from tempest.lib import exceptions
class AccessRuleCreateErrorException(exceptions.TempestException):
    message = 'Access rule %(access)s failed to create and is in ERROR state.'