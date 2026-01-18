import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _print_instances(instances, is_admin=False):
    for instance in instances:
        setattr(instance, 'flavor_id', instance.flavor['id'])
        if hasattr(instance, 'volume'):
            setattr(instance, 'size', instance.volume['size'])
        else:
            setattr(instance, 'size', '-')
        if not hasattr(instance, 'region'):
            setattr(instance, 'region', '')
        if hasattr(instance, 'datastore'):
            if instance.datastore.get('version'):
                setattr(instance, 'datastore_version', instance.datastore['version'])
            setattr(instance, 'datastore', instance.datastore['type'])
    fields = ['id', 'name', 'datastore', 'datastore_version', 'status', 'flavor_id', 'size', 'region']
    if is_admin:
        fields.append('tenant_id')
    utils.print_list(instances, fields)