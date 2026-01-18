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
class CommandLineGetter(object):

    def get_region(self, params):
        if not params.get('region', None):
            prop = self.cls.find_property('region_name')
            params['region'] = propget.get(prop, choices=boto.ec2.regions)

    def get_zone(self, params):
        if not params.get('zone', None):
            prop = StringProperty(name='zone', verbose_name='EC2 Availability Zone', choices=self.ec2.get_all_zones)
            params['zone'] = propget.get(prop)

    def get_name(self, params):
        if not params.get('name', None):
            prop = self.cls.find_property('name')
            params['name'] = propget.get(prop)

    def get_size(self, params):
        if not params.get('size', None):
            prop = IntegerProperty(name='size', verbose_name='Size (GB)')
            params['size'] = propget.get(prop)

    def get_mount_point(self, params):
        if not params.get('mount_point', None):
            prop = self.cls.find_property('mount_point')
            params['mount_point'] = propget.get(prop)

    def get_device(self, params):
        if not params.get('device', None):
            prop = self.cls.find_property('device')
            params['device'] = propget.get(prop)

    def get(self, cls, params):
        self.cls = cls
        self.get_region(params)
        self.ec2 = params['region'].connect()
        self.get_zone(params)
        self.get_name(params)
        self.get_size(params)
        self.get_mount_point(params)
        self.get_device(params)