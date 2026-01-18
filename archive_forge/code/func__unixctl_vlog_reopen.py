import datetime
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import ovs.dirs
import ovs.unixctl
import ovs.util
@staticmethod
def _unixctl_vlog_reopen(conn, unused_argv, unused_aux):
    if Vlog.__log_file:
        Vlog.reopen_log_file()
        conn.reply(None)
    else:
        conn.reply('Logging to file not configured')