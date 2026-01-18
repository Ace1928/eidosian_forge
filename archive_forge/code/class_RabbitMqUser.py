from __future__ import absolute_import, division, print_function
import ansible_collections.community.rabbitmq.plugins.module_utils.version as Version
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.collections import count
class RabbitMqUser(object):

    def __init__(self, module, username, password, tags, permissions, topic_permissions, node, bulk_permissions=False):
        self.module = module
        self.username = username
        self.password = password or ''
        self.node = node
        self.tags = list() if not tags else tags.replace(' ', '').split(',')
        self.permissions = as_permission_dict(permissions)
        self.topic_permissions = as_topic_permission_dict(topic_permissions)
        self.bulk_permissions = bulk_permissions
        self.existing_tags = None
        self.existing_permissions = dict()
        self.existing_topic_permissions = dict()
        self._rabbitmqctl = module.get_bin_path('rabbitmqctl', True)
        self._version = self._check_version()

    def _check_version(self):
        """Get the version of the RabbitMQ server."""
        version = self._rabbitmq_version_post_3_7(fail_on_error=False)
        if not version:
            version = self._rabbitmq_version_pre_3_7(fail_on_error=False)
        if not version:
            self.module.fail_json(msg='Could not determine the version of the RabbitMQ server.')
        return version

    def _fail(self, msg, stop_execution=False):
        if stop_execution:
            self.module.fail_json(msg=msg)
        return None

    def _rabbitmq_version_post_3_7(self, fail_on_error=False):
        """Use the JSON formatter to get a machine readable output of the version.

        At this point we do not know which RabbitMQ server version we are dealing with and which
        version of `rabbitmqctl` we are using, so we will try to use the JSON formatter and see
        what happens. In some versions of
        """

        def int_list_to_str(ints):
            return ''.join([chr(i) for i in ints])
        rc, output, err = self._exec(['status', '--formatter', 'json'], check_rc=False)
        if rc != 0:
            return self._fail(msg='Could not parse the version of the RabbitMQ server, because `rabbitmqctl status` returned no output.', stop_execution=fail_on_error)
        try:
            status_json = json.loads(output)
            if 'rabbitmq_version' in status_json:
                return Version.StrictVersion(status_json['rabbitmq_version'])
            for application in status_json.get('running_applications', list()):
                if application[0] == 'rabbit':
                    if isinstance(application[1][0], int):
                        return Version.StrictVersion(int_list_to_str(application[2]))
                    else:
                        return Version.StrictVersion(application[1])
            return self._fail(msg='Could not find RabbitMQ version of `rabbitmqctl status` command.', stop_execution=fail_on_error)
        except ValueError as e:
            return self._fail(msg='Could not parse output of `rabbitmqctl status` as JSON: {exc}.'.format(exc=repr(e)), stop_execution=fail_on_error)

    def _rabbitmq_version_pre_3_7(self, fail_on_error=False):
        """Get the version of the RabbitMQ Server.

        Before version 3.7.6 the `rabbitmqctl` utility did not support the
        `--formatter` flag, so the output has to be parsed using regular expressions.
        """
        version_reg_ex = '{rabbit,\\"RabbitMQ\\",\\"([0-9]+\\.[0-9]+\\.[0-9]+)\\"}'
        rc, output, err = self._exec(['status'], check_rc=False)
        if rc != 0:
            if fail_on_error:
                self.module.fail_json(msg='Could not parse the version of the RabbitMQ server, because `rabbitmqctl status` returned no output.')
            else:
                return None
        reg_ex_res = re.search(version_reg_ex, output, re.IGNORECASE)
        if not reg_ex_res:
            return self._fail(msg='Could not parse the version of the RabbitMQ server from the output of `rabbitmqctl status` command: {output}.'.format(output=output), stop_execution=fail_on_error)
        try:
            return Version.StrictVersion(reg_ex_res.group(1))
        except ValueError as e:
            return self._fail(msg='Could not parse the version of the RabbitMQ server: {exc}.'.format(exc=repr(e)), stop_execution=fail_on_error)

    def _exec(self, args, check_rc=True):
        """Execute a command using the `rabbitmqctl` utility.

        By default the _exec call will cause the module to fail, if the error code is non-zero. If the `check_rc`
        flag is set to False, then the exit_code, stdout and stderr will be returned to the calling function to
        perform whatever error handling it needs.

        :param args: the arguments to pass to the `rabbitmqctl` utility
        :param check_rc: when set to True, fail if the utility's exit code is non-zero
        :return: the output of the command or all the outputs plus the error code in case of error
        """
        cmd = [self._rabbitmqctl, '-q']
        if self.node:
            cmd.extend(['-n', self.node])
        rc, out, err = self.module.run_command(cmd + args)
        if check_rc and rc != 0:
            user_error_msg_regex = '(Only root or .* .* run rabbitmqctl)'
            user_error_msg = re.search(user_error_msg_regex, out)
            if user_error_msg:
                self.module.fail_json(msg='Wrong user used to run the `rabbitmqctl` utility: {err}'.format(err=user_error_msg.group(1)))
            else:
                self.module.fail_json(msg='rabbitmqctl exited with non-zero code: {err}'.format(err=err), rc=rc, stdout=out)
        return out if check_rc else (rc, out, err)

    def get(self):
        """Retrieves the list of registered users from the node.

        If the user is already present, the node will also be queried for the user's permissions and topic
        permissions.
        If the version of the node is >= 3.7.6 the JSON formatter will be used, otherwise the plaintext will be
        parsed.
        """
        if self._version >= Version.StrictVersion('3.7.6'):
            users = dict([(user_entry['user'], user_entry['tags']) for user_entry in json.loads(self._exec(['list_users', '--formatter', 'json']))])
        else:
            users = self._exec(['list_users'])

            def process_tags(tags):
                if not tags:
                    return list()
                return tags.replace('[', '').replace(']', '').replace(' ', '').strip('\t').split(',')
            users_and_tags = [user_entry.split('\t') for user_entry in users.strip().split('\n')]
            users = dict()
            for user_parts in users_and_tags:
                users[user_parts[0]] = process_tags(user_parts[1]) if len(user_parts) > 1 else []
        self.existing_tags = users.get(self.username, list())
        self.existing_permissions = self._get_permissions() if self.username in users else dict()
        self.existing_topic_permissions = self._get_topic_permissions() if self.username in users else dict()
        return self.username in users

    def _get_permissions(self):
        """Get permissions of the user from RabbitMQ."""
        if self._version >= Version.StrictVersion('3.7.6'):
            permissions = json.loads(self._exec(['list_user_permissions', self.username, '--formatter', 'json']))
        else:
            output = self._exec(['list_user_permissions', self.username]).strip().split('\n')
            perms_out = [perm.split('\t') for perm in output if perm.strip()]
            perms_out = [perm for perm in perms_out if perm != ['vhost', 'configure', 'write', 'read']]
            permissions = list()
            for vhost, configure, write, read in perms_out:
                permissions.append(dict(vhost=vhost, configure=configure, write=write, read=read))
        if self.bulk_permissions:
            return as_permission_dict(permissions)
        else:
            return only(first(self.permissions.keys()), as_permission_dict(permissions))

    def _get_topic_permissions(self):
        """Get topic permissions of the user from RabbitMQ."""
        if self._version < Version.StrictVersion('3.7.0'):
            return dict()
        if self._version >= Version.StrictVersion('3.7.6'):
            permissions = json.loads(self._exec(['list_user_topic_permissions', self.username, '--formatter', 'json']))
        else:
            output = self._exec(['list_user_topic_permissions', self.username]).strip().split('\n')
            perms_out = [perm.split('\t') for perm in output if perm.strip()]
            permissions = list()
            for vhost, exchange, write, read in perms_out:
                permissions.append(dict(vhost=vhost, exchange=exchange, write=write, read=read))
        return as_topic_permission_dict(permissions)

    def check_password(self):
        """Return `True` if the user can authenticate successfully."""
        rc, out, err = self._exec(['authenticate_user', self.username, self.password], check_rc=False)
        return rc == 0

    def add(self):
        self._exec(['add_user', self.username, self.password or ''])
        if not self.password:
            self._exec(['clear_password', self.username])

    def delete(self):
        self._exec(['delete_user', self.username])

    def change_password(self):
        if self.password:
            self._exec(['change_password', self.username, self.password])
        else:
            self._exec(['clear_password', self.username])

    def set_tags(self):
        self._exec(['set_user_tags', self.username] + self.tags)

    def set_permissions(self):
        permissions_to_add = list()
        for vhost, permission_dict in self.permissions.items():
            if permission_dict != self.existing_permissions.get(vhost, {}):
                permissions_to_add.append(permission_dict)
        permissions_to_clear = list()
        for vhost in self.existing_permissions.keys():
            if vhost not in self.permissions:
                permissions_to_clear.append(vhost)
        for vhost in permissions_to_clear:
            cmd = 'clear_permissions -p {vhost} {username}'.format(username=self.username, vhost=vhost)
            self._exec(cmd.split(' '))
        for permissions in permissions_to_add:
            cmd = 'set_permissions -p {vhost} {username} {configure} {write} {read}'.format(username=self.username, **permissions)
            self._exec(cmd.split(' '))
        self.existing_permissions = self._get_permissions()

    def set_topic_permissions(self):
        permissions_to_add = list()
        for vhost_exchange, permission_dict in self.topic_permissions.items():
            if permission_dict != self.existing_topic_permissions.get(vhost_exchange, {}):
                permissions_to_add.append(permission_dict)
        permissions_to_clear = list()
        for vhost_exchange in self.existing_topic_permissions.keys():
            if vhost_exchange not in self.topic_permissions:
                permissions_to_clear.append(vhost_exchange)
        for vhost_exchange in permissions_to_clear:
            vhost, exchange = vhost_exchange
            cmd = 'clear_topic_permissions -p {vhost} {username} {exchange}'.format(username=self.username, vhost=vhost, exchange=exchange)
            self._exec(cmd.split(' '))
        for permissions in permissions_to_add:
            cmd = 'set_topic_permissions -p {vhost} {username} {exchange} {write} {read}'.format(username=self.username, **permissions)
            self._exec(cmd.split(' '))
        self.existing_topic_permissions = self._get_topic_permissions()

    def has_tags_modifications(self):
        return set(self.tags) != set(self.existing_tags)

    def has_permissions_modifications(self):
        return self.existing_permissions != self.permissions

    def has_topic_permissions_modifications(self):
        return self.existing_topic_permissions != self.topic_permissions