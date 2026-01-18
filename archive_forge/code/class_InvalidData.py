from tempest.lib import exceptions
class InvalidData(exceptions.TempestException):
    message = 'Provided invalid data: %(message)s'