from __future__ import (absolute_import, division, print_function)
import logging
import os.path
import subprocess
import sys
from configparser import ConfigParser
import ovirtsdk4 as sdk
from bcolors import bcolors
def _validate_connection(self, log, url, username, password, ca):
    conn = None
    try:
        conn = self._connect_sdk(url, username, password, ca)
        dcs_service = conn.system_service().data_centers_service()
        dcs_service.list()
    except Exception as e:
        msg = 'Connection to setup has failed. Please check your credentials: \n URL: ' + url + '\n user: ' + username + '\n CA file: ' + ca
        log.error(msg)
        print('%s%s%s%s' % (FAIL, PREFIX, msg, END))
        log.error('Error: %s', e)
        if conn:
            conn.close()
        return False
    return True