from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.backup.types import BackupTargetType
class BackupDriver(BaseDriver):
    """
    A base BackupDriver class to derive from

    This class is always subclassed by a specific driver.
    """
    connectionCls = ConnectionUserAndKey
    name = None
    website = None

    def __init__(self, key, secret=None, secure=True, host=None, port=None, **kwargs):
        """
        :param    key: API key or username to used (required)
        :type     key: ``str``

        :param    secret: Secret password to be used (required)
        :type     secret: ``str``

        :param    secure: Whether to use HTTPS or HTTP. Note: Some providers
                only support HTTPS, and it is on by default.
        :type     secure: ``bool``

        :param    host: Override hostname used for connections.
        :type     host: ``str``

        :param    port: Override port used for connections.
        :type     port: ``int``

        :return: ``None``
        """
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, **kwargs)

    def get_supported_target_types(self):
        """
        Get a list of backup target types this driver supports

        :return: ``list`` of :class:``BackupTargetType``
        """
        raise NotImplementedError('get_supported_target_types not implemented for this driver')

    def list_targets(self):
        """
        List all backuptargets

        :rtype: ``list`` of :class:`.BackupTarget`
        """
        raise NotImplementedError('list_targets not implemented for this driver')

    def create_target(self, name, address, type=BackupTargetType.VIRTUAL, extra=None):
        """
        Creates a new backup target

        :param name: Name of the target
        :type name: ``str``

        :param address: Hostname, FQDN, IP, file path etc.
        :type address: ``str``

        :param type: Backup target type (Physical, Virtual, ...).
        :type type: :class:`BackupTargetType`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`.BackupTarget`
        """
        raise NotImplementedError('create_target not implemented for this driver')

    def create_target_from_node(self, node, type=BackupTargetType.VIRTUAL, extra=None):
        """
        Creates a new backup target from an existing node.
        By default, this will use the first public IP of the node

        :param node: The Node to backup
        :type  node: ``Node``

        :param type: Backup target type (Physical, Virtual, ...).
        :type type: :class:`BackupTargetType`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`.BackupTarget`
        """
        return self.create_target(name=node.name, address=node.public_ips[0], type=type, extra=None)

    def create_target_from_storage_container(self, container, type=BackupTargetType.OBJECT, extra=None):
        """
        Creates a new backup target from an existing storage container

        :param node: The Container to backup
        :type  node: ``Container``

        :param type: Backup target type (Physical, Virtual, ...).
        :type type: :class:`BackupTargetType`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`.BackupTarget`
        """
        return self.create_target(name=container.name, address=container.get_cdn_url(), type=type, extra=None)

    def update_target(self, target, name, address, extra):
        """
        Update the properties of a backup target

        :param target: Backup target to update
        :type  target: Instance of :class:`.BackupTarget`

        :param name: Name of the target
        :type name: ``str``

        :param address: Hostname, FQDN, IP, file path etc.
        :type address: ``str``

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`.BackupTarget`
        """
        raise NotImplementedError('update_target not implemented for this driver')

    def delete_target(self, target):
        """
        Delete a backup target

        :param target: Backup target to delete
        :type  target: Instance of :class:`.BackupTarget`
        """
        raise NotImplementedError('delete_target not implemented for this driver')

    def list_recovery_points(self, target, start_date=None, end_date=None):
        """
        List the recovery points available for a target

        :param target: Backup target to delete
        :type  target: Instance of :class:`.BackupTarget`

        :param start_date: The start date to show jobs between (optional)
        :type  start_date: :class:`datetime.datetime`

        :param end_date: The end date to show jobs between (optional)
        :type  end_date: :class:`datetime.datetime``

        :rtype: ``list`` of :class:`.BackupTargetRecoveryPoint`
        """
        raise NotImplementedError('list_recovery_points not implemented for this driver')

    def recover_target(self, target, recovery_point, path=None):
        """
        Recover a backup target to a recovery point

        :param target: Backup target to delete
        :type  target: Instance of :class:`.BackupTarget`

        :param recovery_point: Backup target with the backup data
        :type  recovery_point: Instance of :class:`.BackupTarget`

        :param path: The part of the recovery point to recover (optional)
        :type  path: ``str``

        :rtype: Instance of :class:`.BackupTargetJob`
        """
        raise NotImplementedError('recover_target not implemented for this driver')

    def recover_target_out_of_place(self, target, recovery_point, recovery_target, path=None):
        """
        Recover a backup target to a recovery point out-of-place

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`.BackupTarget`

        :param recovery_point: Backup target with the backup data
        :type  recovery_point: Instance of :class:`.BackupTarget`

        :param recovery_target: Backup target with to recover the data to
        :type  recovery_target: Instance of :class:`.BackupTarget`

        :param path: The part of the recovery point to recover (optional)
        :type  path: ``str``

        :rtype: Instance of :class:`BackupTargetJob`
        """
        raise NotImplementedError('recover_target_out_of_place not implemented for this driver')

    def get_target_job(self, target, id):
        """
        Get a specific backup job by ID

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`.BackupTarget`

        :param id: Backup target with the backup data
        :type  id: Instance of :class:`.BackupTarget`

        :rtype: :class:`BackupTargetJob`
        """
        jobs = self.list_target_jobs(target)
        return list(filter(lambda x: x.id == id, jobs))[0]

    def list_target_jobs(self, target):
        """
        List the backup jobs on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`.BackupTarget`

        :rtype: ``list`` of :class:`.BackupTargetJob`
        """
        raise NotImplementedError('list_target_jobs not implemented for this driver')

    def create_target_job(self, target, extra=None):
        """
        Create a new backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`.BackupTarget`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`BackupTargetJob`
        """
        raise NotImplementedError('create_target_job not implemented for this driver')

    def resume_target_job(self, job):
        """
        Resume a suspended backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`.BackupTarget`

        :param job: Backup target job to resume
        :type  job: Instance of :class:`.BackupTargetJob`

        :rtype: ``bool``
        """
        raise NotImplementedError('resume_target_job not implemented for this driver')

    def suspend_target_job(self, job):
        """
        Suspend a running backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`.BackupTarget`

        :param job: Backup target job to suspend
        :type  job: Instance of :class:`.BackupTargetJob`

        :rtype: ``bool``
        """
        raise NotImplementedError('suspend_target_job not implemented for this driver')

    def cancel_target_job(self, job):
        """
        Cancel a backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`.BackupTarget`

        :param job: Backup target job to cancel
        :type  job: Instance of :class:`.BackupTargetJob`

        :rtype: ``bool``
        """
        raise NotImplementedError('cancel_target_job not implemented for this driver')