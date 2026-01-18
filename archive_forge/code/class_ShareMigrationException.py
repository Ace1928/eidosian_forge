from tempest.lib import exceptions
class ShareMigrationException(exceptions.TempestException):
    message = 'Share %(share_id)s failed to migrate from host %(src)s to host %(dest)s.'