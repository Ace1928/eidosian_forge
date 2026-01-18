from collections import namedtuple
import logging
import sys
from cliff import app
from cliff import command
from cliff import commandmanager
from cliff import complete
from cliff import help
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import session
import barbicanclient
from barbicanclient._i18n import _LW
from barbicanclient import client
def _get_endpoint_filter_kwargs(self, args):
    endpoint_filter_keys = ('interface', 'service_type', 'service_name', 'barbican_api_version', 'region_name')
    kwargs = dict(((key, getattr(args, key)) for key in endpoint_filter_keys if getattr(args, key, None)))
    if 'barbican_api_version' in kwargs:
        kwargs['version'] = kwargs.pop('barbican_api_version')
    return kwargs