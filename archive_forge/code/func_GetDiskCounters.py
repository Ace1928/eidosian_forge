from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import locale
import os
import struct
import sys
import six
from gslib.utils.constants import WINDOWS_1252
def GetDiskCounters():
    """Retrieves disk I/O statistics for all disks.

  Adapted from the psutil module's psutil._pslinux.disk_io_counters:
    http://code.google.com/p/psutil/source/browse/trunk/psutil/_pslinux.py

  Originally distributed under under a BSD license.
  Original Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola.

  Returns:
    A dictionary containing disk names mapped to the disk counters from
    /disk/diskstats.
  """
    sector_size = 512
    partitions = []
    with open('/proc/partitions', 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            _, _, _, name = line.split()
            if name[-1].isdigit():
                partitions.append(name)
    retdict = {}
    with open('/proc/diskstats', 'r') as f:
        for line in f:
            values = line.split()[:11]
            _, _, name, reads, _, rbytes, rtime, writes, _, wbytes, wtime = values
            if name in partitions:
                rbytes = int(rbytes) * sector_size
                wbytes = int(wbytes) * sector_size
                reads = int(reads)
                writes = int(writes)
                rtime = int(rtime)
                wtime = int(wtime)
                retdict[name] = (reads, writes, rbytes, wbytes, rtime, wtime)
    return retdict