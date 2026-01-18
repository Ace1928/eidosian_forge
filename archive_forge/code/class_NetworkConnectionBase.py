from __future__ import (annotations, absolute_import, division, print_function)
import collections.abc as c
import fcntl
import io
import os
import shlex
import typing as t
from abc import abstractmethod
from functools import wraps
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.playbook.play_context import PlayContext
from ansible.plugins import AnsiblePlugin
from ansible.plugins.become import BecomeBase
from ansible.plugins.shell import ShellBase
from ansible.utils.display import Display
from ansible.plugins.loader import connection_loader, get_shell_plugin
from ansible.utils.path import unfrackpath
class NetworkConnectionBase(ConnectionBase):
    """
    A base class for network-style connections.
    """
    force_persistence = True
    _remote_is_local = True

    def __init__(self, play_context: PlayContext, new_stdin: io.TextIOWrapper | None=None, *args: t.Any, **kwargs: t.Any) -> None:
        super(NetworkConnectionBase, self).__init__(play_context, new_stdin, *args, **kwargs)
        self._messages: list[tuple[str, str]] = []
        self._conn_closed = False
        self._network_os = self._play_context.network_os
        self._local = connection_loader.get('local', play_context, '/dev/null')
        self._local.set_options()
        self._sub_plugin: dict[str, t.Any] = {}
        self._cached_variables = (None, None, None)
        self._ansible_playbook_pid = kwargs.get('ansible_playbook_pid')
        self._update_connection_state()

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            if not name.startswith('_'):
                plugin = self._sub_plugin.get('obj')
                if plugin:
                    method = getattr(plugin, name, None)
                    if method is not None:
                        return method
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    def exec_command(self, cmd: str, in_data: bytes | None=None, sudoable: bool=True) -> tuple[int, bytes, bytes]:
        return self._local.exec_command(cmd, in_data, sudoable)

    def queue_message(self, level: str, message: str) -> None:
        """
        Adds a message to the queue of messages waiting to be pushed back to the controller process.

        :arg level: A string which can either be the name of a method in display, or 'log'. When
            the messages are returned to task_executor, a value of log will correspond to
            ``display.display(message, log_only=True)``, while another value will call ``display.[level](message)``
        """
        self._messages.append((level, message))

    def pop_messages(self) -> list[tuple[str, str]]:
        messages, self._messages = (self._messages, [])
        return messages

    def put_file(self, in_path: str, out_path: str) -> None:
        """Transfer a file from local to remote"""
        return self._local.put_file(in_path, out_path)

    def fetch_file(self, in_path: str, out_path: str) -> None:
        """Fetch a file from remote to local"""
        return self._local.fetch_file(in_path, out_path)

    def reset(self) -> None:
        """
        Reset the connection
        """
        if self._socket_path:
            self.queue_message('vvvv', 'resetting persistent connection for socket_path %s' % self._socket_path)
            self.close()
        self.queue_message('vvvv', 'reset call on connection instance')

    def close(self) -> None:
        self._conn_closed = True
        if self._connected:
            self._connected = False

    def set_options(self, task_keys: dict[str, t.Any] | None=None, var_options: dict[str, t.Any] | None=None, direct: dict[str, t.Any] | None=None) -> None:
        super(NetworkConnectionBase, self).set_options(task_keys=task_keys, var_options=var_options, direct=direct)
        if self.get_option('persistent_log_messages'):
            warning = 'Persistent connection logging is enabled for %s. This will log ALL interactions' % self._play_context.remote_addr
            logpath = getattr(C, 'DEFAULT_LOG_PATH')
            if logpath is not None:
                warning += ' to %s' % logpath
            self.queue_message('warning', '%s and WILL NOT redact sensitive configuration like passwords. USE WITH CAUTION!' % warning)
        if self._sub_plugin.get('obj') and self._sub_plugin.get('type') != 'external':
            try:
                self._sub_plugin['obj'].set_options(task_keys=task_keys, var_options=var_options, direct=direct)
            except AttributeError:
                pass

    def _update_connection_state(self) -> None:
        """
        Reconstruct the connection socket_path and check if it exists

        If the socket path exists then the connection is active and set
        both the _socket_path value to the path and the _connected value
        to True.  If the socket path doesn't exist, leave the socket path
        value to None and the _connected value to False
        """
        ssh = connection_loader.get('ssh', class_only=True)
        control_path = ssh._create_control_path(self._play_context.remote_addr, self._play_context.port, self._play_context.remote_user, self._play_context.connection, self._ansible_playbook_pid)
        tmp_path = unfrackpath(C.PERSISTENT_CONTROL_PATH_DIR)
        socket_path = unfrackpath(control_path % dict(directory=tmp_path))
        if os.path.exists(socket_path):
            self._connected = True
            self._socket_path = socket_path

    def _log_messages(self, message: str) -> None:
        if self.get_option('persistent_log_messages'):
            self.queue_message('log', message)