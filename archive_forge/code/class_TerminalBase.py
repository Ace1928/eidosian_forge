from __future__ import (absolute_import, division, print_function)
import re
from abc import ABC, abstractmethod
from ansible.errors import AnsibleConnectionFailure
class TerminalBase(ABC):
    """
    A base class for implementing cli connections

    .. note:: Unlike most of Ansible, nearly all strings in
        :class:`TerminalBase` plugins are byte strings.  This is because of
        how close to the underlying platform these plugins operate.  Remember
        to mark literal strings as byte string (``b"string"``) and to use
        :func:`~ansible.module_utils.common.text.converters.to_bytes` and
        :func:`~ansible.module_utils.common.text.converters.to_text` to avoid unexpected
        problems.
    """
    terminal_stdout_re = []
    terminal_stderr_re = []
    ansi_re = [re.compile(b'\\x1b\\[\\?1h\\x1b='), re.compile(b'\\x08.'), re.compile(b'\\x1b\\[m')]
    terminal_initial_prompt = None
    terminal_initial_answer = None
    terminal_inital_prompt_newline = True

    def __init__(self, connection):
        self._connection = connection

    def _exec_cli_command(self, cmd, check_rc=True):
        """
        Executes the CLI command on the remote device and returns the output

        :arg cmd: Byte string command to be executed
        """
        return self._connection.exec_command(cmd)

    def _get_prompt(self):
        """
        Returns the current prompt from the device

        :returns: A byte string of the prompt
        """
        return self._connection.get_prompt()

    def on_open_shell(self):
        """Called after the SSH session is established

        This method is called right after the invoke_shell() is called from
        the Paramiko SSHClient instance.  It provides an opportunity to setup
        terminal parameters such as disbling paging for instance.
        """
        pass

    def on_close_shell(self):
        """Called before the connection is closed

        This method gets called once the connection close has been requested
        but before the connection is actually closed.  It provides an
        opportunity to clean up any terminal resources before the shell is
        actually closed
        """
        pass

    def on_become(self, passwd=None):
        """Called when privilege escalation is requested

        :kwarg passwd: String containing the password

        This method is called when the privilege is requested to be elevated
        in the play context by setting become to True.  It is the responsibility
        of the terminal plugin to actually do the privilege escalation such
        as entering `enable` mode for instance
        """
        pass

    def on_unbecome(self):
        """Called when privilege deescalation is requested

        This method is called when the privilege changed from escalated
        (become=True) to non escalated (become=False).  It is the responsibility
        of this method to actually perform the deauthorization procedure
        """
        pass

    def on_authorize(self, passwd=None):
        """Deprecated method for privilege escalation

        :kwarg passwd: String containing the password
        """
        return self.on_become(passwd)

    def on_deauthorize(self):
        """Deprecated method for privilege deescalation
        """
        return self.on_unbecome()