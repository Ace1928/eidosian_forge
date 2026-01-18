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
class ConnectionBase(AnsiblePlugin):
    """
    A base class for connections to contain common code.
    """
    has_pipelining = False
    has_native_async = False
    always_pipeline_modules = False
    has_tty = True
    module_implementation_preferences = ('',)
    allow_executable = True
    supports_persistence = False
    force_persistence = False
    default_user: str | None = None

    def __init__(self, play_context: PlayContext, new_stdin: io.TextIOWrapper | None=None, shell: ShellBase | None=None, *args: t.Any, **kwargs: t.Any) -> None:
        super(ConnectionBase, self).__init__()
        if not hasattr(self, '_play_context'):
            self._play_context = play_context
        if not hasattr(self, '__new_stdin'):
            self.__new_stdin = new_stdin
        if not hasattr(self, '_display'):
            self._display = display
        self.success_key = None
        self.prompt = None
        self._connected = False
        self._socket_path: str | None = None
        self._shell = shell
        if not self._shell:
            shell_type = play_context.shell if play_context.shell else getattr(self, '_shell_type', None)
            self._shell = get_shell_plugin(shell_type=shell_type, executable=self._play_context.executable)
        self.become: BecomeBase | None = None

    @property
    def _new_stdin(self) -> io.TextIOWrapper | None:
        display.deprecated("The connection's stdin object is deprecated. Call display.prompt_until(msg) instead.", version='2.19')
        return self.__new_stdin

    def set_become_plugin(self, plugin: BecomeBase) -> None:
        self.become = plugin

    @property
    def connected(self) -> bool:
        """Read-only property holding whether the connection to the remote host is active or closed."""
        return self._connected

    @property
    def socket_path(self) -> str | None:
        """Read-only property holding the connection socket path for this remote host"""
        return self._socket_path

    @staticmethod
    def _split_ssh_args(argstring: str) -> list[str]:
        """
        Takes a string like '-o Foo=1 -o Bar="foo bar"' and returns a
        list ['-o', 'Foo=1', '-o', 'Bar=foo bar'] that can be added to
        the argument list. The list will not contain any empty elements.
        """
        return [to_text(x.strip()) for x in shlex.split(argstring) if x.strip()]

    @property
    @abstractmethod
    def transport(self) -> str:
        """String used to identify this Connection class from other classes"""
        pass

    @abstractmethod
    def _connect(self: T) -> T:
        """Connect to the host we've been initialized with"""

    @ensure_connect
    @abstractmethod
    def exec_command(self, cmd: str, in_data: bytes | None=None, sudoable: bool=True) -> tuple[int, bytes, bytes]:
        """Run a command on the remote host.

        :arg cmd: byte string containing the command
        :kwarg in_data: If set, this data is passed to the command's stdin.
            This is used to implement pipelining.  Currently not all
            connection plugins implement pipelining.
        :kwarg sudoable: Tell the connection plugin if we're executing
            a command via a privilege escalation mechanism.  This may affect
            how the connection plugin returns data.  Note that not all
            connections can handle privilege escalation.
        :returns: a tuple of (return code, stdout, stderr)  The return code is
            an int while stdout and stderr are both byte strings.

        When a command is executed, it goes through multiple commands to get
        there.  It looks approximately like this::

            [LocalShell] ConnectionCommand [UsersLoginShell (*)] ANSIBLE_SHELL_EXECUTABLE [(BecomeCommand ANSIBLE_SHELL_EXECUTABLE)] Command
        :LocalShell: Is optional.  It is run locally to invoke the
            ``Connection Command``.  In most instances, the
            ``ConnectionCommand`` can be invoked directly instead.  The ssh
            connection plugin which can have values that need expanding
            locally specified via ssh_args is the sole known exception to
            this.  Shell metacharacters in the command itself should be
            processed on the remote machine, not on the local machine so no
            shell is needed on the local machine.  (Example, ``/bin/sh``)
        :ConnectionCommand: This is the command that connects us to the remote
            machine to run the rest of the command.  ``ansible_user``,
            ``ansible_ssh_host`` and so forth are fed to this piece of the
            command to connect to the correct host (Examples ``ssh``,
            ``chroot``)
        :UsersLoginShell: This shell may or may not be created depending on
            the ConnectionCommand used by the connection plugin.  This is the
            shell that the ``ansible_user`` has configured as their login
            shell.  In traditional UNIX parlance, this is the last field of
            a user's ``/etc/passwd`` entry   We do not specifically try to run
            the ``UsersLoginShell`` when we connect.  Instead it is implicit
            in the actions that the ``ConnectionCommand`` takes when it
            connects to a remote machine.  ``ansible_shell_type`` may be set
            to inform ansible of differences in how the ``UsersLoginShell``
            handles things like quoting if a shell has different semantics
            than the Bourne shell.
        :ANSIBLE_SHELL_EXECUTABLE: This is the shell set via the inventory var
            ``ansible_shell_executable`` or via
            ``constants.DEFAULT_EXECUTABLE`` if the inventory var is not set.
            We explicitly invoke this shell so that we have predictable
            quoting rules at this point.  ``ANSIBLE_SHELL_EXECUTABLE`` is only
            settable by the user because some sudo setups may only allow
            invoking a specific shell.  (For instance, ``/bin/bash`` may be
            allowed but ``/bin/sh``, our default, may not).  We invoke this
            twice, once after the ``ConnectionCommand`` and once after the
            ``BecomeCommand``.  After the ConnectionCommand, this is run by
            the ``UsersLoginShell``.  After the ``BecomeCommand`` we specify
            that the ``ANSIBLE_SHELL_EXECUTABLE`` is being invoked directly.
        :BecomeComand ANSIBLE_SHELL_EXECUTABLE: Is the command that performs
            privilege escalation.  Setting this up is performed by the action
            plugin prior to running ``exec_command``. So we just get passed
            :param:`cmd` which has the BecomeCommand already added.
            (Examples: sudo, su)  If we have a BecomeCommand then we will
            invoke a ANSIBLE_SHELL_EXECUTABLE shell inside of it so that we
            have a consistent view of quoting.
        :Command: Is the command we're actually trying to run remotely.
            (Examples: mkdir -p $HOME/.ansible, python $HOME/.ansible/tmp-script-file)
        """
        pass

    @ensure_connect
    @abstractmethod
    def put_file(self, in_path: str, out_path: str) -> None:
        """Transfer a file from local to remote"""
        pass

    @ensure_connect
    @abstractmethod
    def fetch_file(self, in_path: str, out_path: str) -> None:
        """Fetch a file from remote to local; callers are expected to have pre-created the directory chain for out_path"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Terminate the connection"""
        pass

    def connection_lock(self) -> None:
        f = self._play_context.connection_lockfd
        display.vvvv('CONNECTION: pid %d waiting for lock on %d' % (os.getpid(), f), host=self._play_context.remote_addr)
        fcntl.lockf(f, fcntl.LOCK_EX)
        display.vvvv('CONNECTION: pid %d acquired lock on %d' % (os.getpid(), f), host=self._play_context.remote_addr)

    def connection_unlock(self) -> None:
        f = self._play_context.connection_lockfd
        fcntl.lockf(f, fcntl.LOCK_UN)
        display.vvvv('CONNECTION: pid %d released lock on %d' % (os.getpid(), f), host=self._play_context.remote_addr)

    def reset(self) -> None:
        display.warning('Reset is not implemented for this connection')

    def update_vars(self, variables: dict[str, t.Any]) -> None:
        """
        Adds 'magic' variables relating to connections to the variable dictionary provided.
        In case users need to access from the play, this is a legacy from runner.
        """
        for varname in C.COMMON_CONNECTION_VARS:
            value = None
            if varname in variables:
                continue
            elif 'password' in varname or 'passwd' in varname:
                continue
            elif varname == 'ansible_connection':
                value = self._load_name
            elif varname == 'ansible_shell_type' and self._shell:
                value = self._shell._load_name
            else:
                options = C.config.get_plugin_options_from_var('connection', self._load_name, varname)
                if options:
                    value = self.get_option(options[0])
                elif 'become' not in varname:
                    for prop, var_list in C.MAGIC_VARIABLE_MAPPING.items():
                        if varname in var_list:
                            try:
                                value = getattr(self._play_context, prop)
                                break
                            except AttributeError:
                                continue
            if value is not None:
                display.debug('Set connection var {0} to {1}'.format(varname, value))
                variables[varname] = value