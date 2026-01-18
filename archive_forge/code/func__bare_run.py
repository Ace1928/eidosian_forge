from __future__ import (annotations, absolute_import, division, print_function)
import collections.abc as c
import errno
import fcntl
import hashlib
import io
import os
import pty
import re
import shlex
import subprocess
import time
import typing as t
from functools import wraps
from ansible.errors import (
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.compat import selectors
from ansible.module_utils.six import PY3, text_type, binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import BOOLEANS, boolean
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath, makedirs_safe
def _bare_run(self, cmd: list[bytes], in_data: bytes | None, sudoable: bool=True, checkrc: bool=True) -> tuple[int, bytes, bytes]:
    """
        Starts the command and communicates with it until it ends.
        """
    display_cmd = u' '.join((shlex.quote(to_text(c)) for c in cmd))
    display.vvv(u'SSH: EXEC {0}'.format(display_cmd), host=self.host)
    p = None
    if isinstance(cmd, (text_type, binary_type)):
        cmd = to_bytes(cmd)
    else:
        cmd = list(map(to_bytes, cmd))
    conn_password = self.get_option('password') or self._play_context.password
    if not in_data:
        try:
            master, slave = pty.openpty()
            if PY3 and conn_password:
                p = subprocess.Popen(cmd, stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=self.sshpass_pipe)
            else:
                p = subprocess.Popen(cmd, stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdin = os.fdopen(master, 'wb', 0)
            os.close(slave)
        except (OSError, IOError):
            p = None
    if not p:
        try:
            if PY3 and conn_password:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=self.sshpass_pipe)
            else:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdin = p.stdin
        except (OSError, IOError) as e:
            raise AnsibleError('Unable to execute ssh command line on a controller due to: %s' % to_native(e))
    if conn_password:
        os.close(self.sshpass_pipe[0])
        try:
            os.write(self.sshpass_pipe[1], to_bytes(conn_password) + b'\n')
        except OSError as e:
            if e.errno != errno.EPIPE or p.poll() is None:
                raise
        os.close(self.sshpass_pipe[1])
    states = ['awaiting_prompt', 'awaiting_escalation', 'ready_to_send', 'awaiting_exit']
    state = states.index('ready_to_send')
    if to_bytes(self.get_option('ssh_executable')) in cmd and sudoable:
        prompt = getattr(self.become, 'prompt', None)
        if prompt:
            state = states.index('awaiting_prompt')
            display.debug(u'Initial state: %s: %s' % (states[state], to_text(prompt)))
        elif self.become and self.become.success:
            state = states.index('awaiting_escalation')
            display.debug(u'Initial state: %s: %s' % (states[state], to_text(self.become.success)))
    b_stdout = b_stderr = b''
    b_tmp_stdout = b_tmp_stderr = b''
    self._flags = dict(become_prompt=False, become_success=False, become_error=False, become_nopasswd_error=False)
    timeout = 2 + self.get_option('timeout')
    for fd in (p.stdout, p.stderr):
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
    selector = selectors.DefaultSelector()
    selector.register(p.stdout, selectors.EVENT_READ)
    selector.register(p.stderr, selectors.EVENT_READ)
    if states[state] == 'ready_to_send' and in_data:
        self._send_initial_data(stdin, in_data, p)
        state += 1
    try:
        while True:
            poll = p.poll()
            events = selector.select(timeout)
            if not events:
                if state <= states.index('awaiting_escalation'):
                    if poll is not None:
                        break
                    self._terminate_process(p)
                    raise AnsibleError('Timeout (%ds) waiting for privilege escalation prompt: %s' % (timeout, to_native(b_stdout)))
            for key, event in events:
                if key.fileobj == p.stdout:
                    b_chunk = p.stdout.read()
                    if b_chunk == b'':
                        selector.unregister(p.stdout)
                        timeout = 1
                    b_tmp_stdout += b_chunk
                    display.debug(u'stdout chunk (state=%s):\n>>>%s<<<\n' % (state, to_text(b_chunk)))
                elif key.fileobj == p.stderr:
                    b_chunk = p.stderr.read()
                    if b_chunk == b'':
                        selector.unregister(p.stderr)
                    b_tmp_stderr += b_chunk
                    display.debug('stderr chunk (state=%s):\n>>>%s<<<\n' % (state, to_text(b_chunk)))
            if state < states.index('ready_to_send'):
                if b_tmp_stdout:
                    b_output, b_unprocessed = self._examine_output('stdout', states[state], b_tmp_stdout, sudoable)
                    b_stdout += b_output
                    b_tmp_stdout = b_unprocessed
                if b_tmp_stderr:
                    b_output, b_unprocessed = self._examine_output('stderr', states[state], b_tmp_stderr, sudoable)
                    b_stderr += b_output
                    b_tmp_stderr = b_unprocessed
            else:
                b_stdout += b_tmp_stdout
                b_stderr += b_tmp_stderr
                b_tmp_stdout = b_tmp_stderr = b''
            if states[state] == 'awaiting_prompt':
                if self._flags['become_prompt']:
                    display.debug(u'Sending become_password in response to prompt')
                    become_pass = self.become.get_option('become_pass', playcontext=self._play_context)
                    stdin.write(to_bytes(become_pass, errors='surrogate_or_strict') + b'\n')
                    stdin.flush()
                    self._flags['become_prompt'] = False
                    state += 1
                elif self._flags['become_success']:
                    state += 1
            if states[state] == 'awaiting_escalation':
                if self._flags['become_success']:
                    display.vvv(u'Escalation succeeded')
                    self._flags['become_success'] = False
                    state += 1
                elif self._flags['become_error']:
                    display.vvv(u'Escalation failed')
                    self._terminate_process(p)
                    self._flags['become_error'] = False
                    raise AnsibleError('Incorrect %s password' % self.become.name)
                elif self._flags['become_nopasswd_error']:
                    display.vvv(u'Escalation requires password')
                    self._terminate_process(p)
                    self._flags['become_nopasswd_error'] = False
                    raise AnsibleError('Missing %s password' % self.become.name)
                elif self._flags['become_prompt']:
                    display.vvv(u'Escalation prompt repeated')
                    self._terminate_process(p)
                    self._flags['become_prompt'] = False
                    raise AnsibleError('Incorrect %s password' % self.become.name)
            if states[state] == 'ready_to_send':
                if in_data:
                    self._send_initial_data(stdin, in_data, p)
                state += 1
            if poll is not None:
                if not selector.get_map() or not events:
                    break
                timeout = 0
                continue
            elif not selector.get_map():
                p.wait()
                break
    finally:
        selector.close()
        stdin.close()
        p.stdout.close()
        p.stderr.close()
    if self.get_option('host_key_checking'):
        if cmd[0] == b'sshpass' and p.returncode == 6:
            raise AnsibleError("Using a SSH password instead of a key is not possible because Host Key checking is enabled and sshpass does not support this.  Please add this host's fingerprint to your known_hosts file to manage this host.")
    controlpersisterror = b'Bad configuration option: ControlPersist' in b_stderr or b'unknown configuration option: ControlPersist' in b_stderr
    if p.returncode != 0 and controlpersisterror:
        raise AnsibleError('using -c ssh on certain older ssh versions may not support ControlPersist, set ANSIBLE_SSH_ARGS="" (or ssh_args in [ssh_connection] section of the config file) before running again')
    controlpersist_broken_pipe = b'mux_client_hello_exchange: write packet: Broken pipe' in b_stderr
    if p.returncode == 255:
        additional = to_native(b_stderr)
        if controlpersist_broken_pipe:
            raise AnsibleControlPersistBrokenPipeError('Data could not be sent because of ControlPersist broken pipe: %s' % additional)
        elif in_data and checkrc:
            raise AnsibleConnectionFailure('Data could not be sent to remote host "%s". Make sure this host can be reached over ssh: %s' % (self.host, additional))
    return (p.returncode, b_stdout, b_stderr)