from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def _register_using_dbus(self, was_registered, username, password, auto_attach, activationkey, org_id, consumer_type, consumer_name, consumer_id, force_register, environment, release):
    """
            Register using D-Bus (connecting to the rhsm service)

            Raises:
              * Exception - if error occurs during the D-Bus communication
        """
    import dbus
    SUBSCRIPTION_MANAGER_LOCALE = 'C'
    REGISTRATION_TIMEOUT = 600

    def str2int(s, default=0):
        try:
            return int(s)
        except ValueError:
            return default
    distro_id = distro.id()
    distro_version_parts = distro.version_parts()
    distro_version = tuple((str2int(p) for p in distro_version_parts))
    if distro_id == 'fedora' or distro_version[0] >= 7:
        cmd = ['systemctl', 'stop', 'rhsm']
        self.module.run_command(cmd, check_rc=True, expand_user_and_vars=False)
    dbus_force_option_works = False
    if distro_id == 'rhel' and (distro_version[0] == 8 and distro_version[1] >= 8 or (distro_version[0] == 9 and distro_version[1] >= 2) or distro_version[0] > 9):
        dbus_force_option_works = True
    if force_register and (not dbus_force_option_works) and was_registered:
        self.unregister()
    register_opts = {}
    if consumer_type:

        def supports_option_consumer_type():
            if distro_id == 'fedora':
                return True
            if distro_id == 'rhel' and (distro_version[0] == 9 and distro_version[1] >= 2 or distro_version[0] >= 10):
                return True
            if distro_id == 'centos' and distro_version[0] >= 9:
                return True
            return False
        consumer_type_key = 'type'
        if supports_option_consumer_type():
            consumer_type_key = 'consumer_type'
        register_opts[consumer_type_key] = consumer_type
    if consumer_name:
        register_opts['name'] = consumer_name
    if consumer_id:
        register_opts['consumerid'] = consumer_id
    if environment:

        def supports_option_environments():
            if distro_id == 'fedora':
                return True
            if distro_id == 'rhel' and (distro_version[0] == 8 and distro_version[1] >= 6 or distro_version[0] >= 9):
                return True
            if distro_id == 'centos' and (distro_version[0] == 8 and (distro_version[1] >= 6 or distro_version_parts[1] == '') or distro_version[0] >= 9):
                return True
            return False
        environment_key = 'environment'
        if supports_option_environments():
            environment_key = 'environments'
        register_opts[environment_key] = environment
    if force_register and dbus_force_option_works and was_registered:
        register_opts['force'] = True
    register_opts = dbus.Dictionary(register_opts, signature='sv', variant_level=1)
    connection_opts = {}
    connection_opts = dbus.Dictionary(connection_opts, signature='sv', variant_level=1)
    bus = dbus.SystemBus()
    register_server = bus.get_object('com.redhat.RHSM1', '/com/redhat/RHSM1/RegisterServer')
    address = register_server.Start(SUBSCRIPTION_MANAGER_LOCALE, dbus_interface='com.redhat.RHSM1.RegisterServer')
    try:
        self.module.debug('Connecting to the private DBus')
        private_bus = dbus.connection.Connection(address)
        try:
            if activationkey:
                args = (org_id, [activationkey], register_opts, connection_opts, SUBSCRIPTION_MANAGER_LOCALE)
                private_bus.call_blocking('com.redhat.RHSM1', '/com/redhat/RHSM1/Register', 'com.redhat.RHSM1.Register', 'RegisterWithActivationKeys', 'sasa{sv}a{sv}s', args, timeout=REGISTRATION_TIMEOUT)
            else:
                args = (org_id or '', username, password, register_opts, connection_opts, SUBSCRIPTION_MANAGER_LOCALE)
                private_bus.call_blocking('com.redhat.RHSM1', '/com/redhat/RHSM1/Register', 'com.redhat.RHSM1.Register', 'Register', 'sssa{sv}a{sv}s', args, timeout=REGISTRATION_TIMEOUT)
        except dbus.exceptions.DBusException as e:
            if e.get_dbus_name() == 'org.freedesktop.DBus.Error.NoReply':
                if not self.is_registered():
                    raise
            else:
                raise
    finally:
        self.module.debug('Shutting down private DBus instance')
        register_server.Stop(SUBSCRIPTION_MANAGER_LOCALE, dbus_interface='com.redhat.RHSM1.RegisterServer')
    self.module.run_command([SUBMAN_CMD, 'refresh'], check_rc=True, expand_user_and_vars=False)
    if auto_attach:
        args = [SUBMAN_CMD, 'attach', '--auto']
        self.module.run_command(args, check_rc=True, expand_user_and_vars=False)
    if release:
        args = [SUBMAN_CMD, 'release', '--set', release]
        self.module.run_command(args, check_rc=True, expand_user_and_vars=False)