from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.common.google import GoogleResponse, GoogleBaseConnection
from libcloud.utils.iso8601 import parse_date
class GCEBackupDriver(BackupDriver):
    name = 'Google Compute Engine Backup Driver'
    website = 'http://cloud.google.com/'
    connectionCls = GCEConnection

    def __init__(self, user_id, key=None, project=None, auth_type=None, scopes=None, credential_file=None, **kwargs):
        """
        :param  user_id: The email address (for service accounts) or Client ID
                         (for installed apps) to be used for authentication.
        :type   user_id: ``str``

        :param  key: The RSA Key (for service accounts) or file path containing
                     key or Client Secret (for installed apps) to be used for
                     authentication.
        :type   key: ``str``

        :keyword  project: Your GCE project name. (required)
        :type     project: ``str``

        :keyword  auth_type: Accepted values are "SA" or "IA" or "GCE"
                             ("Service Account" or "Installed Application" or
                             "GCE" if libcloud is being used on a GCE instance
                             with service account enabled).
                             If not supplied, auth_type will be guessed based
                             on value of user_id or if the code is being
                             executed in a GCE instance.
        :type     auth_type: ``str``

        :keyword  scopes: List of authorization URLs. Default is empty and
                          grants read/write to Compute, Storage, DNS.
        :type     scopes: ``list``

        :keyword  credential_file: Path to file for caching authentication
                                   information used by GCEConnection.
        :type     credential_file: ``str``
        """
        if not project:
            raise ValueError('Project name must be specified using "project" keyword.')
        self.auth_type = auth_type
        self.project = project
        self.scopes = scopes
        self.credential_file = credential_file or '~/.gce_libcloud_auth' + '.' + self.project
        super().__init__(user_id, key, **kwargs)
        self.base_path = '/compute/{}/projects/{}'.format(API_VERSION, self.project)

    def get_supported_target_types(self):
        """
        Get a list of backup target types this driver supports

        :return: ``list`` of :class:``BackupTargetType``
        """
        return [BackupTargetType.VOLUME]

    def list_targets(self):
        """
        List all backuptargets

        :rtype: ``list`` of :class:`BackupTarget`
        """
        raise NotImplementedError('list_targets not implemented for this driver')

    def create_target(self, name, address, type=BackupTargetType.VOLUME, extra=None):
        """
        Creates a new backup target

        :param name: Name of the target
        :type name: ``str``

        :param address: The volume ID.
        :type address: ``str``

        :param type: Backup target type (Physical, Virtual, ...).
        :type type: :class:`BackupTargetType`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`BackupTarget`
        """
        return self.ex_get_target_by_source(address)

    def create_target_from_node(self, node, type=BackupTargetType.VIRTUAL, extra=None):
        """
        Creates a new backup target from an existing node

        :param node: The Node to backup
        :type  node: ``Node``

        :param type: Backup target type (Physical, Virtual, ...).
        :type type: :class:`BackupTargetType`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`BackupTarget`
        """
        disks = node.extra['disks']
        if disks is not None:
            return self.create_target(name=node.name, address=disks[0]['source'], type=BackupTargetType.VOLUME, extra=None)
        else:
            raise RuntimeError('Node does not have any block devices')

    def create_target_from_container(self, container, type=BackupTargetType.OBJECT, extra=None):
        """
        Creates a new backup target from an existing storage container

        :param node: The Container to backup
        :type  node: ``Container``

        :param type: Backup target type (Physical, Virtual, ...).
        :type type: :class:`BackupTargetType`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`BackupTarget`
        """
        raise NotImplementedError('create_target_from_container not implemented for this driver')

    def update_target(self, target, name, address, extra):
        """
        Update the properties of a backup target

        :param target: Backup target to update
        :type  target: Instance of :class:`BackupTarget`

        :param name: Name of the target
        :type name: ``str``

        :param address: Hostname, FQDN, IP, file path etc.
        :type address: ``str``

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`BackupTarget`
        """
        return self.ex_get_target_by_source(address)

    def delete_target(self, target):
        """
        Delete a backup target

        :param target: Backup target to delete
        :type  target: Instance of :class:`BackupTarget`
        """
        raise NotImplementedError('delete_target not implemented for this driver')

    def list_recovery_points(self, target, start_date=None, end_date=None):
        """
        List the recovery points available for a target

        :param target: Backup target to delete
        :type  target: Instance of :class:`BackupTarget`

        :param start_date: The start date to show jobs between (optional)
        :type  start_date: :class:`datetime.datetime`

        :param end_date: The end date to show jobs between (optional)
        :type  end_date: :class:`datetime.datetime``

        :rtype: ``list`` of :class:`BackupTargetRecoveryPoint`
        """
        request = '/global/snapshots'
        response = self.connection.request(request, method='GET').object
        return self._to_recovery_points(response, target)

    def recover_target(self, target, recovery_point, path=None):
        """
        Recover a backup target to a recovery point

        :param target: Backup target to delete
        :type  target: Instance of :class:`BackupTarget`

        :param recovery_point: Backup target with the backup data
        :type  recovery_point: Instance of :class:`BackupTarget`

        :param path: The part of the recovery point to recover (optional)
        :type  path: ``str``

        :rtype: Instance of :class:`BackupTargetJob`
        """
        raise NotImplementedError('recover_target not implemented for this driver')

    def recover_target_out_of_place(self, target, recovery_point, recovery_target, path=None):
        """
        Recover a backup target to a recovery point out-of-place

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :param recovery_point: Backup target with the backup data
        :type  recovery_point: Instance of :class:`BackupTarget`

        :param recovery_target: Backup target with to recover the data to
        :type  recovery_target: Instance of :class:`BackupTarget`

        :param path: The part of the recovery point to recover (optional)
        :type  path: ``str``

        :rtype: Instance of :class:`BackupTargetJob`
        """
        raise NotImplementedError('recover_target_out_of_place not implemented for this driver')

    def get_target_job(self, target, id):
        """
        Get a specific backup job by ID

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :param id: Backup target with the backup data
        :type  id: Instance of :class:`BackupTarget`

        :rtype: :class:`BackupTargetJob`
        """
        jobs = self.list_target_jobs(target)
        return list(filter(lambda x: x.id == id, jobs))[0]

    def list_target_jobs(self, target):
        """
        List the backup jobs on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :rtype: ``list`` of :class:`BackupTargetJob`
        """
        return []

    def create_target_job(self, target, extra=None):
        """
        Create a new backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`BackupTargetJob`
        """
        name = target.name
        request = '/zones/{}/disks/{}/createSnapshot'.format(target.extra['zone'].name, target.name)
        snapshot_data = {'source': target.extra['source']}
        self.connection.async_request(request, method='POST', data=snapshot_data)
        return self._to_job(self.ex_get_snapshot(name), target)

    def resume_target_job(self, target, job):
        """
        Resume a suspended backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :param job: Backup target job to resume
        :type  job: Instance of :class:`BackupTargetJob`

        :rtype: ``bool``
        """
        raise NotImplementedError('resume_target_job not supported for this driver')

    def suspend_target_job(self, target, job):
        """
        Suspend a running backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :param job: Backup target job to suspend
        :type  job: Instance of :class:`BackupTargetJob`

        :rtype: ``bool``
        """
        raise NotImplementedError('suspend_target_job not supported for this driver')

    def cancel_target_job(self, target, job):
        """
        Cancel a backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :param job: Backup target job to cancel
        :type  job: Instance of :class:`BackupTargetJob`

        :rtype: ``bool``
        """
        raise NotImplementedError('cancel_target_job not supported for this driver')

    def _to_recovery_points(self, data, target):
        return [self._to_recovery_point(item, target) for item in data.items]

    def _to_recovery_point(self, item, target):
        id = item.id
        date = parse_date(item.creationTimestamp)
        point = BackupTargetRecoveryPoint(id=id, date=date, target=target, driver=self.connection.driver, extra={'snapshot-id': id})
        return point

    def _to_jobs(self, data, target):
        return [self._to_job(item, target) for item in data.items]

    def _to_job(self, item, target):
        id = item.id
        job = BackupTargetJob(id=id, status=BackupTargetJobStatusType.PENDING, progress=0, target=target, driver=self.connection.driver, extra={})
        return job

    def ex_get_snapshot(self, name):
        request = '/global/snapshots/%s' % name
        response = self.connection.request(request, method='GET').object
        return response

    def ex_get_target_by_source(self, source):
        return BackupTarget(id=source, name=source, address=source, type=BackupTargetType.VOLUME, driver=self.connection.driver, extra={'source': source})