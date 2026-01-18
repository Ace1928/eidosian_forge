from __future__ import print_function
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, ListProperty, ReferenceProperty, CalculatedProperty
from boto.manage.server import Server
from boto.manage import propget
import boto.utils
import boto.ec2
import time
import traceback
from contextlib import closing
import datetime
def _volume_state(self):
    ec2 = self.get_ec2_connection()
    rs = ec2.get_all_volumes([self.volume_id])
    return rs[0].volume_state()