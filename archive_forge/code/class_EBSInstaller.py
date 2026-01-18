import boto
from boto.manage.volume import Volume
from boto.exception import EC2ResponseError
import os, time
from boto.pyami.installers.ubuntu.installer import Installer
from string import Template
import boto
from boto.pyami.scriptbase import ScriptBase
import traceback
import boto
from boto.manage.volume import Volume
import boto
class EBSInstaller(Installer):
    """
    Set up the EBS stuff
    """

    def __init__(self, config_file=None):
        super(EBSInstaller, self).__init__(config_file)
        self.instance_id = boto.config.get('Instance', 'instance-id')
        self.device = boto.config.get('EBS', 'device', '/dev/sdp')
        self.volume_id = boto.config.get('EBS', 'volume_id')
        self.logical_volume_name = boto.config.get('EBS', 'logical_volume_name')
        self.mount_point = boto.config.get('EBS', 'mount_point', '/ebs')

    def attach(self):
        ec2 = boto.connect_ec2()
        if self.logical_volume_name:
            logical_volume = next(Volume.find(name=self.logical_volume_name))
            self.volume_id = logical_volume._volume_id
        volume = ec2.get_all_volumes([self.volume_id])[0]
        while volume.update() != 'available':
            boto.log.info('Volume %s not yet available. Current status = %s.' % (volume.id, volume.status))
            time.sleep(5)
        instance = ec2.get_only_instances([self.instance_id])[0]
        attempt_attach = True
        while attempt_attach:
            try:
                ec2.attach_volume(self.volume_id, self.instance_id, self.device)
                attempt_attach = False
            except EC2ResponseError as e:
                if e.error_code != 'IncorrectState':
                    boto.log.info('Attempt to attach the EBS volume %s to this instance (%s) returned %s. Trying again in a bit.' % (self.volume_id, self.instance_id, e.errors))
                    time.sleep(2)
                else:
                    raise e
        boto.log.info('Attached volume %s to instance %s as device %s' % (self.volume_id, self.instance_id, self.device))
        while not os.path.exists(self.device):
            boto.log.info('%s still does not exist, waiting 2 seconds' % self.device)
            time.sleep(2)

    def make_fs(self):
        boto.log.info('make_fs...')
        has_fs = self.run('fsck %s' % self.device)
        if has_fs != 0:
            self.run('mkfs -t xfs %s' % self.device)

    def create_backup_script(self):
        t = Template(BackupScriptTemplate)
        s = t.substitute(volume_id=self.volume_id, instance_id=self.instance_id, mount_point=self.mount_point)
        fp = open('/usr/local/bin/ebs_backup', 'w')
        fp.write(s)
        fp.close()
        self.run('chmod +x /usr/local/bin/ebs_backup')

    def create_backup_cleanup_script(self, use_tag_based_cleanup=False):
        fp = open('/usr/local/bin/ebs_backup_cleanup', 'w')
        if use_tag_based_cleanup:
            fp.write(TagBasedBackupCleanupScript)
        else:
            fp.write(BackupCleanupScript)
        fp.close()
        self.run('chmod +x /usr/local/bin/ebs_backup_cleanup')

    def handle_mount_point(self):
        boto.log.info('handle_mount_point')
        if not os.path.isdir(self.mount_point):
            boto.log.info('making directory')
            self.run('mkdir %s' % self.mount_point)
        else:
            boto.log.info('directory exists already')
            self.run('mount -l')
            lines = self.last_command.output.split('\n')
            for line in lines:
                t = line.split()
                if t and t[2] == self.mount_point:
                    if t[0] != self.device:
                        self.run('umount %s' % self.mount_point)
                        self.run('mount %s /tmp' % t[0])
                        break
        self.run('chmod 777 /tmp')
        self.run('mount %s %s' % (self.device, self.mount_point))
        self.run('xfs_growfs %s' % self.mount_point)

    def update_fstab(self):
        f = open('/etc/fstab', 'a')
        f.write('%s\t%s\txfs\tdefaults 0 0\n' % (self.device, self.mount_point))
        f.close()

    def install(self):
        self.attach()
        self.run('apt-get -y install xfsprogs xfsdump')
        self.make_fs()
        self.handle_mount_point()
        self.create_backup_script()
        minute = boto.config.get('EBS', 'backup_cron_minute', '0')
        hour = boto.config.get('EBS', 'backup_cron_hour', '4,16')
        self.add_cron('ebs_backup', '/usr/local/bin/ebs_backup', minute=minute, hour=hour)
        minute = boto.config.get('EBS', 'backup_cleanup_cron_minute')
        hour = boto.config.get('EBS', 'backup_cleanup_cron_hour')
        if minute is not None and hour is not None:
            use_tag_based_cleanup = boto.config.has_option('EBS', 'use_tag_based_snapshot_cleanup')
            self.create_backup_cleanup_script(use_tag_based_cleanup)
            self.add_cron('ebs_backup_cleanup', '/usr/local/bin/ebs_backup_cleanup', minute=minute, hour=hour)
        self.update_fstab()

    def main(self):
        if not os.path.exists(self.device):
            self.install()
        else:
            boto.log.info('Device %s is already attached, skipping EBS Installer' % self.device)