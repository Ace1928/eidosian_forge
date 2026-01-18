from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
class Imgadm(object):

    def __init__(self, module):
        self.module = module
        self.params = module.params
        self.cmd = module.get_bin_path('imgadm', required=True)
        self.changed = False
        self.uuid = module.params['uuid']
        if self.params['state'] in ['present', 'imported', 'updated']:
            self.present = True
        else:
            self.present = False
        if self.uuid and self.uuid != '*':
            if not re.match('^[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$', self.uuid, re.IGNORECASE):
                module.fail_json(msg='Provided value for uuid option is not a valid UUID.')

    def errmsg(self, stderr):
        match = re.match('^imgadm .*?: error \\(\\w+\\): (.*): .*', stderr)
        if match:
            return match.groups()[0]
        else:
            return 'Unexpected failure'

    def update_images(self):
        if self.uuid == '*':
            cmd = '{0} update'.format(self.cmd)
        else:
            cmd = '{0} update {1}'.format(self.cmd, self.uuid)
        rc, stdout, stderr = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to update images: {0}'.format(self.errmsg(stderr)))
        self.changed = True

    def manage_sources(self):
        force = self.params['force']
        source = self.params['source']
        imgtype = self.params['type']
        cmd = '{0} sources'.format(self.cmd)
        if force:
            cmd += ' -f'
        if self.present:
            cmd = '{0} -a {1} -t {2}'.format(cmd, source, imgtype)
            rc, stdout, stderr = self.module.run_command(cmd)
            if rc != 0:
                self.module.fail_json(msg='Failed to add source: {0}'.format(self.errmsg(stderr)))
            regex = 'Already have "{0}" image source "{1}", no change'.format(imgtype, source)
            if re.match(regex, stdout):
                self.changed = False
            regex = 'Added "%s" image source "%s"' % (imgtype, source)
            if re.match(regex, stdout):
                self.changed = True
        else:
            cmd += ' -d %s' % source
            rc, stdout, stderr = self.module.run_command(cmd)
            if rc != 0:
                self.module.fail_json(msg='Failed to remove source: {0}'.format(self.errmsg(stderr)))
            regex = 'Do not have image source "%s", no change' % source
            if re.match(regex, stdout):
                self.changed = False
            regex = 'Deleted ".*" image source "%s"' % source
            if re.match(regex, stdout):
                self.changed = True

    def manage_images(self):
        pool = self.params['pool']
        state = self.params['state']
        if state == 'vacuumed':
            cmd = '{0} vacuum -f'.format(self.cmd)
            rc, stdout, stderr = self.module.run_command(cmd)
            if rc != 0:
                self.module.fail_json(msg='Failed to vacuum images: {0}'.format(self.errmsg(stderr)))
            elif stdout == '':
                self.changed = False
            else:
                self.changed = True
        if self.present:
            cmd = '{0} import -P {1} -q {2}'.format(self.cmd, pool, self.uuid)
            rc, stdout, stderr = self.module.run_command(cmd)
            if rc != 0:
                self.module.fail_json(msg='Failed to import image: {0}'.format(self.errmsg(stderr)))
            regex = 'Image {0} \\(.*\\) is already installed, skipping'.format(self.uuid)
            if re.match(regex, stdout):
                self.changed = False
            regex = '.*ActiveImageNotFound.*'
            if re.match(regex, stderr):
                self.changed = False
            regex = 'Imported image {0}.*'.format(self.uuid)
            if re.match(regex, stdout.splitlines()[-1]):
                self.changed = True
        else:
            cmd = '{0} delete -P {1} {2}'.format(self.cmd, pool, self.uuid)
            rc, stdout, stderr = self.module.run_command(cmd)
            regex = '.*ImageNotInstalled.*'
            if re.match(regex, stderr):
                self.changed = False
            regex = 'Deleted image {0}'.format(self.uuid)
            if re.match(regex, stdout):
                self.changed = True