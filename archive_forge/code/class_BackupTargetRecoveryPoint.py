from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.backup.types import BackupTargetType
class BackupTargetRecoveryPoint:
    """
    A backup target recovery point
    """

    def __init__(self, id, date, target, driver, extra=None):
        """
        :param id: Job id
        :type id: ``str``

        :param date: The date taken
        :type date: :class:`datetime.datetime`

        :param target: BackupTarget instance.
        :type target: :class:`.BackupTarget`

        :param driver: BackupDriver instance.
        :type driver: :class:`.BackupDriver`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``
        """
        self.id = str(id) if id else None
        self.date = date
        self.target = target
        self.driver = driver
        self.extra = extra or {}

    def recover(self, path=None):
        """
        Recover this recovery point

        :param path: The part of the recovery point to recover (optional)
        :type  path: ``str``

        :rtype: Instance of :class:`.BackupTargetJob`
        """
        return self.driver.recover_target(target=self.target, recovery_point=self, path=path)

    def recover_to(self, recovery_target, path=None):
        """
        Recover this recovery point out of place

        :param recovery_target: Backup target with to recover the data to
        :type  recovery_target: Instance of :class:`.BackupTarget`

        :param path: The part of the recovery point to recover (optional)
        :type  path: ``str``

        :rtype: Instance of :class:`.BackupTargetJob`
        """
        return self.driver.recover_target_out_of_place(target=self.target, recovery_point=self, recovery_target=recovery_target, path=path)

    def __repr__(self):
        return '<RecoveryPoint: id=%s, date=%s, target=%s, provider=%s ...>' % (self.id, self.date, self.target.id, self.driver.name)