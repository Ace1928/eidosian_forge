from __future__ import absolute_import, division, print_function
import crypt
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
class Homectl(object):
    """#TODO DOC STRINGS"""

    def __init__(self, module):
        self.module = module
        self.state = module.params['state']
        self.name = module.params['name']
        self.password = module.params['password']
        self.storage = module.params['storage']
        self.disksize = module.params['disksize']
        self.resize = module.params['resize']
        self.realname = module.params['realname']
        self.realm = module.params['realm']
        self.email = module.params['email']
        self.location = module.params['location']
        self.iconname = module.params['iconname']
        self.homedir = module.params['homedir']
        self.imagepath = module.params['imagepath']
        self.uid = module.params['uid']
        self.gid = module.params['gid']
        self.umask = module.params['umask']
        self.memberof = module.params['memberof']
        self.skeleton = module.params['skeleton']
        self.shell = module.params['shell']
        self.environment = module.params['environment']
        self.timezone = module.params['timezone']
        self.locked = module.params['locked']
        self.passwordhint = module.params['passwordhint']
        self.sshkeys = module.params['sshkeys']
        self.language = module.params['language']
        self.notbefore = module.params['notbefore']
        self.notafter = module.params['notafter']
        self.mountopts = module.params['mountopts']
        self.result = {}

    def homed_service_active(self):
        is_active = True
        cmd = ['systemctl', 'show', 'systemd-homed.service', '-p', 'ActiveState']
        rc, show_service_stdout, stderr = self.module.run_command(cmd)
        if rc == 0:
            state = show_service_stdout.rsplit('=')[1]
            if state.strip() != 'active':
                is_active = False
        return is_active

    def user_exists(self):
        exists = False
        valid_pw = False
        rc, stdout, stderr = self.get_user_metadata()
        if rc == 0:
            exists = True
            if self.state != 'absent':
                stored_pwhash = json.loads(stdout)['privileged']['hashedPassword'][0]
                if self._check_password(stored_pwhash):
                    valid_pw = True
        return (exists, valid_pw)

    def create_user(self):
        record = self.create_json_record(create=True)
        cmd = [self.module.get_bin_path('homectl', True)]
        cmd.append('create')
        cmd.append('--identity=-')
        return self.module.run_command(cmd, data=record)

    def _hash_password(self, password):
        method = crypt.METHOD_SHA512
        salt = crypt.mksalt(method, rounds=10000)
        pw_hash = crypt.crypt(password, salt)
        return pw_hash

    def _check_password(self, pwhash):
        hash = crypt.crypt(self.password, pwhash)
        return pwhash == hash

    def remove_user(self):
        cmd = [self.module.get_bin_path('homectl', True)]
        cmd.append('remove')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def prepare_modify_user_command(self):
        record = self.create_json_record()
        cmd = [self.module.get_bin_path('homectl', True)]
        cmd.append('update')
        cmd.append(self.name)
        cmd.append('--identity=-')
        if self.disksize and self.resize:
            cmd.append('--and-resize')
            cmd.append('true')
            self.result['changed'] = True
        return (cmd, record)

    def get_user_metadata(self):
        cmd = [self.module.get_bin_path('homectl', True)]
        cmd.append('inspect')
        cmd.append(self.name)
        cmd.append('-j')
        cmd.append('--no-pager')
        rc, stdout, stderr = self.module.run_command(cmd)
        return (rc, stdout, stderr)

    def create_json_record(self, create=False):
        record = {}
        user_metadata = {}
        self.result['changed'] = False
        if not create:
            rc, user_metadata, stderr = self.get_user_metadata()
            user_metadata = json.loads(user_metadata)
            user_metadata.pop('signature', None)
            user_metadata.pop('binding', None)
            user_metadata.pop('status', None)
            user_metadata.pop('lastChangeUSec', None)
            record = user_metadata
        record['userName'] = self.name
        record['secret'] = {'password': [self.password]}
        if create:
            password_hash = self._hash_password(self.password)
            record['privileged'] = {'hashedPassword': [password_hash]}
            self.result['changed'] = True
        if self.uid and self.gid and create:
            record['uid'] = self.uid
            record['gid'] = self.gid
            self.result['changed'] = True
        if self.memberof:
            member_list = list(self.memberof.split(','))
            if member_list != record.get('memberOf', [None]):
                record['memberOf'] = member_list
                self.result['changed'] = True
        if self.realname:
            if self.realname != record.get('realName'):
                record['realName'] = self.realname
                self.result['changed'] = True
        if self.storage and create:
            record['storage'] = self.storage
            self.result['changed'] = True
        if self.homedir and create:
            record['homeDirectory'] = self.homedir
            self.result['changed'] = True
        if self.imagepath and create:
            record['imagePath'] = self.imagepath
            self.result['changed'] = True
        if self.disksize:
            if self.disksize != record.get('diskSize'):
                record['diskSize'] = human_to_bytes(self.disksize)
                self.result['changed'] = True
        if self.realm:
            if self.realm != record.get('realm'):
                record['realm'] = self.realm
                self.result['changed'] = True
        if self.email:
            if self.email != record.get('emailAddress'):
                record['emailAddress'] = self.email
                self.result['changed'] = True
        if self.location:
            if self.location != record.get('location'):
                record['location'] = self.location
                self.result['changed'] = True
        if self.iconname:
            if self.iconname != record.get('iconName'):
                record['iconName'] = self.iconname
                self.result['changed'] = True
        if self.skeleton:
            if self.skeleton != record.get('skeletonDirectory'):
                record['skeletonDirectory'] = self.skeleton
                self.result['changed'] = True
        if self.shell:
            if self.shell != record.get('shell'):
                record['shell'] = self.shell
                self.result['changed'] = True
        if self.umask:
            if self.umask != record.get('umask'):
                record['umask'] = self.umask
                self.result['changed'] = True
        if self.environment:
            if self.environment != record.get('environment', [None]):
                record['environment'] = list(self.environment.split(','))
                self.result['changed'] = True
        if self.timezone:
            if self.timezone != record.get('timeZone'):
                record['timeZone'] = self.timezone
                self.result['changed'] = True
        if self.locked:
            if self.locked != record.get('locked'):
                record['locked'] = self.locked
                self.result['changed'] = True
        if self.passwordhint:
            if self.passwordhint != record.get('privileged', {}).get('passwordHint'):
                record['privileged']['passwordHint'] = self.passwordhint
                self.result['changed'] = True
        if self.sshkeys:
            if self.sshkeys != record.get('privileged', {}).get('sshAuthorizedKeys'):
                record['privileged']['sshAuthorizedKeys'] = list(self.sshkeys.split(','))
                self.result['changed'] = True
        if self.language:
            if self.locked != record.get('preferredLanguage'):
                record['preferredLanguage'] = self.language
                self.result['changed'] = True
        if self.notbefore:
            if self.locked != record.get('notBeforeUSec'):
                record['notBeforeUSec'] = self.notbefore
                self.result['changed'] = True
        if self.notafter:
            if self.locked != record.get('notAfterUSec'):
                record['notAfterUSec'] = self.notafter
                self.result['changed'] = True
        if self.mountopts:
            opts = list(self.mountopts.split(','))
            if 'nosuid' in opts:
                if record.get('mountNoSuid') is not True:
                    record['mountNoSuid'] = True
                    self.result['changed'] = True
            elif record.get('mountNoSuid') is not False:
                record['mountNoSuid'] = False
                self.result['changed'] = True
            if 'nodev' in opts:
                if record.get('mountNoDevices') is not True:
                    record['mountNoDevices'] = True
                    self.result['changed'] = True
            elif record.get('mountNoDevices') is not False:
                record['mountNoDevices'] = False
                self.result['changed'] = True
            if 'noexec' in opts:
                if record.get('mountNoExecute') is not True:
                    record['mountNoExecute'] = True
                    self.result['changed'] = True
            elif record.get('mountNoExecute') is not False:
                record['mountNoExecute'] = False
                self.result['changed'] = True
        return jsonify(record)