import logging
import os
import urllib.parse
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import units
import requests
from requests import adapters
from requests.packages.urllib3.util import retry
import glance_store
from glance_store import capabilities
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _, _LE
from glance_store import location
def _get_datastore(self, datacenter_path, datastore_name):
    dc_obj = self._get_datacenter(datacenter_path)
    datastore_ret = self.session.invoke_api(vim_util, 'get_object_property', self.session.vim, dc_obj.ref, 'datastore')
    if datastore_ret:
        datastore_refs = datastore_ret.ManagedObjectReference
        for ds_ref in datastore_refs:
            ds_obj = oslo_datastore.get_datastore_by_ref(self.session, ds_ref)
            if ds_obj.name == datastore_name:
                ds_obj.datacenter = dc_obj
                return ds_obj