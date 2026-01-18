from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def _tokenizeCommandStr(commandStr):
    """Tokenizes commandStr into a list of commands and their arguments for
    the run() function. Returns the list."""
    commandPattern = re.compile('^(su|sd|ss|c|l|m|r|g|d|k|w|h|f|s|a|p)')
    commandList = []
    i = 0
    while i < len(commandStr):
        if commandStr[i] in (' ', '\t', '\n', '\r'):
            i += 1
            continue
        mo = commandPattern.match(commandStr[i:])
        if mo is None:
            raise PyAutoGUIException('Invalid command at index %s: %s is not a valid command' % (i, commandStr[i]))
        individualCommand = mo.group(1)
        commandList.append(individualCommand)
        i += len(individualCommand)
        if individualCommand in ('c', 'l', 'm', 'r', 'su', 'sd', 'ss'):
            pass
        elif individualCommand in ('g', 'd'):
            try:
                x = _getNumberToken(commandStr[i:])
                i += len(x)
                comma = _getCommaToken(commandStr[i:])
                i += len(comma)
                y = _getNumberToken(commandStr[i:])
                i += len(y)
            except PyAutoGUIException as excObj:
                indexPart, colon, message = str(excObj).partition(':')
                indexNum = indexPart[len('Invalid command at index '):]
                newIndexNum = int(indexNum) + i
                raise PyAutoGUIException('Invalid command at index %s:%s' % (newIndexNum, message))
            if x.lstrip()[0].isdecimal() and (not y.lstrip()[0].isdecimal()):
                raise PyAutoGUIException('Invalid command at index %s: Y has a +/- but X does not.' % (i - len(y)))
            if not x.lstrip()[0].isdecimal() and y.lstrip()[0].isdecimal():
                raise PyAutoGUIException('Invalid command at index %s: Y does not have a +/- but X does.' % (i - len(y)))
            commandList.append(x.lstrip())
            commandList.append(y.lstrip())
        elif individualCommand in ('s', 'p'):
            try:
                num = _getNumberToken(commandStr[i:])
                i += len(num)
            except PyAutoGUIException as excObj:
                indexPart, colon, message = str(excObj).partition(':')
                indexNum = indexPart[len('Invalid command at index '):]
                newIndexNum = int(indexNum) + i
                raise PyAutoGUIException('Invalid command at index %s:%s' % (newIndexNum, message))
            commandList.append(num.lstrip())
        elif individualCommand in ('k', 'w', 'h', 'a'):
            try:
                quotedString = _getQuotedStringToken(commandStr[i:])
                i += len(quotedString)
            except PyAutoGUIException as excObj:
                indexPart, colon, message = str(excObj).partition(':')
                indexNum = indexPart[len('Invalid command at index '):]
                newIndexNum = int(indexNum) + i
                raise PyAutoGUIException('Invalid command at index %s:%s' % (newIndexNum, message))
            commandList.append(quotedString[1:-1].lstrip())
        elif individualCommand == 'f':
            try:
                numberOfLoops = _getNumberToken(commandStr[i:])
                i += len(numberOfLoops)
                subCommandStr = _getParensCommandStrToken(commandStr[i:])
                i += len(subCommandStr)
            except PyAutoGUIException as excObj:
                indexPart, colon, message = str(excObj).partition(':')
                indexNum = indexPart[len('Invalid command at index '):]
                newIndexNum = int(indexNum) + i
                raise PyAutoGUIException('Invalid command at index %s:%s' % (newIndexNum, message))
            commandList.append(numberOfLoops.lstrip())
            subCommandStr = subCommandStr.lstrip()[1:-1]
            commandList.append(_tokenizeCommandStr(subCommandStr))
    return commandList