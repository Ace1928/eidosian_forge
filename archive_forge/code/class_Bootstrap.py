import os
import boto
from boto.utils import get_instance_metadata, get_instance_userdata
from boto.pyami.config import Config, BotoConfigPath
from boto.pyami.scriptbase import ScriptBase
import time
class Bootstrap(ScriptBase):
    """
    The Bootstrap class is instantiated and run as part of the PyAMI
    instance initialization process.  The methods in this class will
    be run from the rc.local script of the instance and will be run
    as the root user.

    The main purpose of this class is to make sure the boto distribution
    on the instance is the one required.
    """

    def __init__(self):
        self.working_dir = '/mnt/pyami'
        self.write_metadata()
        super(Bootstrap, self).__init__()

    def write_metadata(self):
        fp = open(os.path.expanduser(BotoConfigPath), 'w')
        fp.write('[Instance]\n')
        inst_data = get_instance_metadata()
        for key in inst_data:
            fp.write('%s = %s\n' % (key, inst_data[key]))
        user_data = get_instance_userdata()
        fp.write('\n%s\n' % user_data)
        fp.write('[Pyami]\n')
        fp.write('working_dir = %s\n' % self.working_dir)
        fp.close()
        boto.config = Config()
        boto.init_logging()

    def create_working_dir(self):
        boto.log.info('Working directory: %s' % self.working_dir)
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

    def load_boto(self):
        update = boto.config.get('Boto', 'boto_update', 'svn:HEAD')
        if update.startswith('svn'):
            if update.find(':') >= 0:
                method, version = update.split(':')
                version = '-r%s' % version
            else:
                version = '-rHEAD'
            location = boto.config.get('Boto', 'boto_location', '/usr/local/boto')
            self.run('svn update %s %s' % (version, location))
        elif update.startswith('git'):
            location = boto.config.get('Boto', 'boto_location', '/usr/share/python-support/python-boto/boto')
            num_remaining_attempts = 10
            while num_remaining_attempts > 0:
                num_remaining_attempts -= 1
                try:
                    self.run('git pull', cwd=location)
                    num_remaining_attempts = 0
                except Exception as e:
                    boto.log.info('git pull attempt failed with the following exception. Trying again in a bit. %s', e)
                    time.sleep(2)
            if update.find(':') >= 0:
                method, version = update.split(':')
            else:
                version = 'master'
            self.run('git checkout %s' % version, cwd=location)
        else:
            self.run('rm /usr/local/lib/python2.5/site-packages/boto')
            self.run('easy_install %s' % update)

    def fetch_s3_file(self, s3_file):
        try:
            from boto.utils import fetch_file
            f = fetch_file(s3_file)
            path = os.path.join(self.working_dir, s3_file.split('/')[-1])
            open(path, 'w').write(f.read())
        except:
            boto.log.exception('Problem Retrieving file: %s' % s3_file)
            path = None
        return path

    def load_packages(self):
        package_str = boto.config.get('Pyami', 'packages')
        if package_str:
            packages = package_str.split(',')
            for package in packages:
                package = package.strip()
                if package.startswith('s3:'):
                    package = self.fetch_s3_file(package)
                if package:
                    if not package.endswith('.py'):
                        self.run('easy_install -Z %s' % package, exit_on_error=False)

    def main(self):
        self.create_working_dir()
        self.load_boto()
        self.load_packages()
        self.notify('Bootstrap Completed for %s' % boto.config.get_instance('instance-id'))