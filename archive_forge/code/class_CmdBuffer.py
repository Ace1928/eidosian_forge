import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
class CmdBuffer(list):

    def __init__(self, delim='\n'):
        super(CmdBuffer, self).__init__()
        self.delim = delim

    def __lshift__(self, value):
        self.append(value)

    def __str__(self):
        return self.delim.join(self)