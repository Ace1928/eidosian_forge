import os
import zlib
import time  # noqa
import logging
import numpy as np
class DoActionTag(Tag):

    def __init__(self, action='stop'):
        Tag.__init__(self)
        self.tagtype = 12
        self.actions = [action]

    def append(self, action):
        self.actions.append(action)

    def process_tag(self):
        bb = bytes()
        for action in self.actions:
            action = action.lower()
            if action == 'stop':
                bb += '\x07'.encode('ascii')
            elif action == 'play':
                bb += '\x06'.encode('ascii')
            else:
                logger.warning('unknown action: %s' % action)
        bb += int2uint8(0)
        self.bytes = bb