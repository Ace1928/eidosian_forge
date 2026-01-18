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
def _unixctl_vlog_list(conn, unused_argv, unused_aux):
    conn.reply(Vlog.get_levels())