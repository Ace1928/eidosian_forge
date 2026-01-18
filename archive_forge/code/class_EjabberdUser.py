from __future__ import absolute_import, division, print_function
import syslog
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
class EjabberdUser(object):
    """ This object represents a user resource for an ejabberd server.   The
    object manages user creation and deletion using ejabberdctl.  The following
    commands are currently supported:
        * ejabberdctl register
        * ejabberdctl unregister
    """

    def __init__(self, module):
        self.module = module
        self.logging = module.params.get('logging')
        self.state = module.params.get('state')
        self.host = module.params.get('host')
        self.user = module.params.get('username')
        self.pwd = module.params.get('password')
        self.runner = CmdRunner(module, command='ejabberdctl', arg_formats=dict(cmd=cmd_runner_fmt.as_list(), host=cmd_runner_fmt.as_list(), user=cmd_runner_fmt.as_list(), pwd=cmd_runner_fmt.as_list()), check_rc=False)

    @property
    def changed(self):
        """ This method will check the current user and see if the password has
        changed.   It will return True if the user does not match the supplied
        credentials and False if it does not
        """
        return self.run_command('check_password', 'user host pwd', lambda rc, out, err: bool(rc))

    @property
    def exists(self):
        """ This method will check to see if the supplied username exists for
        host specified.  If the user exists True is returned, otherwise False
        is returned
        """
        return self.run_command('check_account', 'user host', lambda rc, out, err: not bool(rc))

    def log(self, entry):
        """ This method will log information to the local syslog facility """
        if self.logging:
            syslog.openlog('ansible-%s' % self.module._name)
            syslog.syslog(syslog.LOG_NOTICE, entry)

    def run_command(self, cmd, options, process=None):
        """ This method will run the any command specified and return the
        returns using the Ansible common module
        """

        def _proc(*a):
            return a
        if process is None:
            process = _proc
        with self.runner('cmd ' + options, output_process=process) as ctx:
            res = ctx.run(cmd=cmd, host=self.host, user=self.user, pwd=self.pwd)
            self.log('command: %s' % ' '.join(ctx.run_info['cmd']))
        return res

    def update(self):
        """ The update method will update the credentials for the user provided
        """
        return self.run_command('change_password', 'user host pwd')

    def create(self):
        """ The create method will create a new user on the host with the
        password provided
        """
        return self.run_command('register', 'user host pwd')

    def delete(self):
        """ The delete method will delete the user from the host
        """
        return self.run_command('unregister', 'user host')