import boto.ec2
from boto.mashups.iobject import IObject
from boto.pyami.config import BotoConfigPath, Config
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, BooleanProperty, CalculatedProperty
from boto.manage import propget
from boto.ec2.zone import Zone
from boto.ec2.keypair import KeyPair
import os, time
from contextlib import closing
from boto.exception import EC2ResponseError
from boto.compat import six, StringIO
class Bundler(object):

    def __init__(self, server, uname='root'):
        from boto.manage.cmdshell import SSHClient
        self.server = server
        self.uname = uname
        self.ssh_client = SSHClient(server, uname=uname)

    def copy_x509(self, key_file, cert_file):
        print('\tcopying cert and pk over to /mnt directory on server')
        self.ssh_client.open_sftp()
        path, name = os.path.split(key_file)
        self.remote_key_file = '/mnt/%s' % name
        self.ssh_client.put_file(key_file, self.remote_key_file)
        path, name = os.path.split(cert_file)
        self.remote_cert_file = '/mnt/%s' % name
        self.ssh_client.put_file(cert_file, self.remote_cert_file)
        print('...complete!')

    def bundle_image(self, prefix, size, ssh_key):
        command = ''
        if self.uname != 'root':
            command = 'sudo '
        command += 'ec2-bundle-vol '
        command += '-c %s -k %s ' % (self.remote_cert_file, self.remote_key_file)
        command += '-u %s ' % self.server._reservation.owner_id
        command += '-p %s ' % prefix
        command += '-s %d ' % size
        command += '-d /mnt '
        if self.server.instance_type == 'm1.small' or self.server.instance_type == 'c1.medium':
            command += '-r i386'
        else:
            command += '-r x86_64'
        return command

    def upload_bundle(self, bucket, prefix, ssh_key):
        command = ''
        if self.uname != 'root':
            command = 'sudo '
        command += 'ec2-upload-bundle '
        command += '-m /mnt/%s.manifest.xml ' % prefix
        command += '-b %s ' % bucket
        command += '-a %s ' % self.server.ec2.aws_access_key_id
        command += '-s %s ' % self.server.ec2.aws_secret_access_key
        return command

    def bundle(self, bucket=None, prefix=None, key_file=None, cert_file=None, size=None, ssh_key=None, fp=None, clear_history=True):
        iobject = IObject()
        if not bucket:
            bucket = iobject.get_string('Name of S3 bucket')
        if not prefix:
            prefix = iobject.get_string('Prefix for AMI file')
        if not key_file:
            key_file = iobject.get_filename('Path to RSA private key file')
        if not cert_file:
            cert_file = iobject.get_filename('Path to RSA public cert file')
        if not size:
            size = iobject.get_int('Size (in MB) of bundled image')
        if not ssh_key:
            ssh_key = self.server.get_ssh_key_file()
        self.copy_x509(key_file, cert_file)
        if not fp:
            fp = StringIO()
        fp.write('sudo mv %s /mnt/boto.cfg; ' % BotoConfigPath)
        fp.write('mv ~/.ssh/authorized_keys /mnt/authorized_keys; ')
        if clear_history:
            fp.write('history -c; ')
        fp.write(self.bundle_image(prefix, size, ssh_key))
        fp.write('; ')
        fp.write(self.upload_bundle(bucket, prefix, ssh_key))
        fp.write('; ')
        fp.write('sudo mv /mnt/boto.cfg %s; ' % BotoConfigPath)
        fp.write('mv /mnt/authorized_keys ~/.ssh/authorized_keys')
        command = fp.getvalue()
        print('running the following command on the remote server:')
        print(command)
        t = self.ssh_client.run(command)
        print('\t%s' % t[0])
        print('\t%s' % t[1])
        print('...complete!')
        print('registering image...')
        self.image_id = self.server.ec2.register_image(name=prefix, image_location='%s/%s.manifest.xml' % (bucket, prefix))
        return self.image_id