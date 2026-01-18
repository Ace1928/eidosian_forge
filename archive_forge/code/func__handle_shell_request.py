from copy import copy
import logging
import os.path
import sys
import paramiko
from os_ken import version
from os_ken.lib import hub
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.root import RootCmd
from os_ken.services.protocols.bgp.operator.internal_api import InternalApi
def _handle_shell_request(self):
    LOG.info('session start')
    chan = self.transport.accept(20)
    if not chan:
        LOG.info('transport.accept timed out')
        return
    self.chan = chan
    self.buf = []
    self.curpos = 0
    self.history = []
    self.histindex = 0
    self.prompted = True
    self.chan.send(self.WELCOME)
    self._startnewline()
    while self.is_connected:
        c = self.chan.recv(1)
        c = c.decode()
        if len(c) == 0:
            break
        LOG.debug('ord:%d, hex:0x%x', ord(c), ord(c))
        self.promptlen = len(self.PROMPT) if self.prompted else 0
        if c == '?':
            cmpleter = self.root
            cmds = [tkn.strip() for tkn in ''.join(self.buf).split()]
            for i, cmd in enumerate(cmds):
                subcmds = cmpleter.subcommands
                matches = [x for x in subcmds.keys() if x.startswith(cmd)]
                if len(matches) == 1:
                    cmpled_cmd = matches[0]
                    cmpleter = subcmds[cmpled_cmd](self.api)
            result, cmd = cmpleter('?')
            result = result.value.replace('\n', '\n\r').rstrip()
            self.prompted = False
            buf = copy(self.buf)
            self._startnewline(buf=result)
            self.prompted = True
            self._startnewline(buf=buf)
        elif self._is_echoable(c):
            self.buf.insert(self.curpos - self.promptlen, c)
            self.curpos += 1
            self._refreshline()
        elif self._is_esc(c):
            self._handle_esc_seq()
        elif self._is_eof(c):
            self.end_session()
        elif self._is_curmov(c):
            if c == chr(1):
                self._movcursor(self.promptlen)
            elif c == chr(2):
                self._movcursor(self.curpos - 1)
            elif c == chr(5):
                self._movcursor(self.promptlen + len(self.buf))
            elif c == chr(6):
                self._movcursor(self.curpos + 1)
            else:
                LOG.error('unknown cursor move cmd.')
                continue
        elif self._is_hist(c):
            if c == chr(16):
                self._lookup_hist_up()
            elif c == chr(14):
                self._lookup_hist_down()
        elif self._is_del(c):
            if c == chr(4):
                if self.curpos < self.promptlen + len(self.buf):
                    self.buf.pop(self.curpos - self.promptlen)
                    self._refreshline()
            elif c == chr(8) or c == chr(127):
                if self.curpos > self.promptlen:
                    self.buf.pop(self.curpos - self.promptlen - 1)
                    self.curpos -= 1
                    self._refreshline()
            elif c == chr(21):
                self._clearline()
            elif c == chr(23):
                pos = self.curpos - self.promptlen
                i = pos
                flag = False
                for c in reversed(self.buf[:pos]):
                    if flag and c == ' ':
                        break
                    if c != ' ':
                        flag = True
                    i -= 1
                del self.buf[i:pos]
                self.curpos = self.promptlen + i
                self._refreshline()
            elif c == chr(12):
                self._clearscreen()
        elif self._is_cmpl(c):
            self._do_cmpl(self.buf)
        elif self._is_enter(c):
            if len(''.join(self.buf).strip()) != 0:
                cmds = self._do_cmpl(self.buf, is_exec=True)
                if cmds:
                    self.history.insert(0, self.buf)
                    self.histindex = 0
                    self._execute_cmd(cmds)
                else:
                    LOG.debug('no command is interpreted. just start a new line.')
                    self._startnewline()
            else:
                LOG.debug('blank buf is detected. just start a new line.')
                self._startnewline()
        LOG.debug('curpos: %d, buf: %s, prompted: %s', self.curpos, self.buf, self.prompted)
    LOG.info('session end')