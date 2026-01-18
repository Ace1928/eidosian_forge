from tempest.lib import exceptions
class ShareBuildErrorException(exceptions.TempestException):
    message = 'Share %(share)s failed to build and is in ERROR status.'