import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
def _handle_csi(self, csi, params, command):
    try:
        if command == 'm':
            p = params.split(';')[0]
            if not p:
                p = '0'
            if p in ANSI_FG:
                self.cursor.char.fg = p
            elif p in ANSI_BG:
                self.cursor.char.bg = p
            elif p == ANSI_RESET:
                self.cursor.char.reset()
            elif p in ANSI_STYLES:
                style = ANSI_STYLES[p]
                off = style.startswith('/')
                if off:
                    style = style[1:]
                self.cursor.char[style] = not off
        else:
            abcd = {'A': 'cursor_up', 'B': 'cursor_down', 'C': 'cursor_right', 'D': 'cursor_left'}
            cursor_fn = abcd.get(command)
            if cursor_fn:
                getattr(self, cursor_fn)(int(params) if params else 1)
            elif command == 'J':
                p = params.split(';')[0]
                p = int(p) if p else 0
                self.erase_screen(p)
            elif command == 'K':
                p = params.split(';')[0]
                p = int(p) if p else 0
                self.erase_line(p)
            elif command == 'L':
                p = int(params) if params else 1
                self.insert_lines(p)
            elif command in 'Hf':
                p = params.split(';')
                if len(p) == 2:
                    p = (int(p[0]), int(p[1]))
                elif len(p) == 1:
                    p = (int(p[0]), 1)
                else:
                    p = (1, 1)
                self.cursor_postion(*p)
    except Exception:
        pass