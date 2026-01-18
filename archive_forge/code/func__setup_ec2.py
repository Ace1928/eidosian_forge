import boto.ec2
from boto.mashups.iobject import IObject
from boto.pyami.config import BotoConfigPath, Config
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, BooleanProperty, CalculatedProperty
from boto.manage import propget
from boto.ec2.zone import Zone
from boto.ec2.keypair import KeyPair
import os, time
from contextlib import closing
from boto.exception import EC2ResponseError
from boto.compat import six, StringIO
def _setup_ec2(self):
    if self.ec2 and self._instance and self._reservation:
        return
    if self.id:
        if self.region_name:
            for region in boto.ec2.regions():
                if region.name == self.region_name:
                    self.ec2 = region.connect()
                    if self.instance_id and (not self._instance):
                        try:
                            rs = self.ec2.get_all_reservations([self.instance_id])
                            if len(rs) >= 1:
                                for instance in rs[0].instances:
                                    if instance.id == self.instance_id:
                                        self._reservation = rs[0]
                                        self._instance = instance
                        except EC2ResponseError:
                            pass