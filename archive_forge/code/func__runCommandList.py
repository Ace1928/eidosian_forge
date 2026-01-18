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
def _runCommandList(commandList, _ssCount):
    global PAUSE
    i = 0
    while i < len(commandList):
        command = commandList[i]
        if command == 'c':
            click(button=PRIMARY)
        elif command == 'l':
            click(button=LEFT)
        elif command == 'm':
            click(button=MIDDLE)
        elif command == 'r':
            click(button=RIGHT)
        elif command == 'su':
            scroll(1)
        elif command == 'sd':
            scroll(-1)
        elif command == 'ss':
            screenshot('screenshot%s.png' % _ssCount[0])
            _ssCount[0] += 1
        elif command == 's':
            sleep(float(commandList[i + 1]))
            i += 1
        elif command == 'p':
            PAUSE = float(commandList[i + 1])
            i += 1
        elif command == 'g':
            if commandList[i + 1][0] in ('+', '-') and commandList[i + 2][0] in ('+', '-'):
                move(int(commandList[i + 1]), int(commandList[i + 2]))
            else:
                moveTo(int(commandList[i + 1]), int(commandList[i + 2]))
            i += 2
        elif command == 'd':
            if commandList[i + 1][0] in ('+', '-') and commandList[i + 2][0] in ('+', '-'):
                drag(int(commandList[i + 1]), int(commandList[i + 2]))
            else:
                dragTo(int(commandList[i + 1]), int(commandList[i + 2]))
            i += 2
        elif command == 'k':
            press(commandList[i + 1])
            i += 1
        elif command == 'w':
            write(commandList[i + 1])
            i += 1
        elif command == 'h':
            hotkey(*commandList[i + 1].replace(' ', '').split(','))
            i += 1
        elif command == 'a':
            alert(commandList[i + 1])
            i += 1
        elif command == 'f':
            for j in range(int(commandList[i + 1])):
                _runCommandList(commandList[i + 2], _ssCount)
            i += 2
        i += 1