import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _command(self, name, *args):
    if self.state not in Commands[name]:
        self.literal = None
        raise self.error('command %s illegal in state %s, only allowed in states %s' % (name, self.state, ', '.join(Commands[name])))
    for typ in ('OK', 'NO', 'BAD'):
        if typ in self.untagged_responses:
            del self.untagged_responses[typ]
    if 'READ-ONLY' in self.untagged_responses and (not self.is_readonly):
        raise self.readonly('mailbox status changed to READ-ONLY')
    tag = self._new_tag()
    name = bytes(name, self._encoding)
    data = tag + b' ' + name
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, str):
            arg = bytes(arg, self._encoding)
        data = data + b' ' + arg
    literal = self.literal
    if literal is not None:
        self.literal = None
        if type(literal) is type(self._command):
            literator = literal
        else:
            literator = None
            data = data + bytes(' {%s}' % len(literal), self._encoding)
    if __debug__:
        if self.debug >= 4:
            self._mesg('> %r' % data)
        else:
            self._log('> %r' % data)
    try:
        self.send(data + CRLF)
    except OSError as val:
        raise self.abort('socket error: %s' % val)
    if literal is None:
        return tag
    while 1:
        while self._get_response():
            if self.tagged_commands[tag]:
                return tag
        if literator:
            literal = literator(self.continuation_response)
        if __debug__:
            if self.debug >= 4:
                self._mesg('write literal size %s' % len(literal))
        try:
            self.send(literal)
            self.send(CRLF)
        except OSError as val:
            raise self.abort('socket error: %s' % val)
        if not literator:
            break
    return tag