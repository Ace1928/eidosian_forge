import datetime
import errno
import logging
import os
import subprocess
import sys
class LogStream(object):

    def write(self, data):
        LOG.info(data)