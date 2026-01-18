import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def _ip_gen():
    for host in netaddr.IPRange(self.start_ip, self.end_ip):
        yield host