import boto.ec2
from boto.sdb.db.property import StringProperty, IntegerProperty
from boto.manage import propget
from boto.compat import six
class BuyReservation(object):

    def get_region(self, params):
        if not params.get('region', None):
            prop = StringProperty(name='region', verbose_name='EC2 Region', choices=boto.ec2.regions)
            params['region'] = propget.get(prop, choices=boto.ec2.regions)

    def get_instance_type(self, params):
        if not params.get('instance_type', None):
            prop = StringProperty(name='instance_type', verbose_name='Instance Type', choices=InstanceTypes)
            params['instance_type'] = propget.get(prop)

    def get_quantity(self, params):
        if not params.get('quantity', None):
            prop = IntegerProperty(name='quantity', verbose_name='Number of Instances')
            params['quantity'] = propget.get(prop)

    def get_zone(self, params):
        if not params.get('zone', None):
            prop = StringProperty(name='zone', verbose_name='EC2 Availability Zone', choices=self.ec2.get_all_zones)
            params['zone'] = propget.get(prop)

    def get(self, params):
        self.get_region(params)
        self.ec2 = params['region'].connect()
        self.get_instance_type(params)
        self.get_zone(params)
        self.get_quantity(params)