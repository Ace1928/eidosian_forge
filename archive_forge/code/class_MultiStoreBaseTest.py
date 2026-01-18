import os
import shutil
import fixtures
from oslo_config import cfg
from oslotest import base
import glance_store as store
from glance_store import location
class MultiStoreBaseTest(base.BaseTestCase):

    def copy_data_file(self, file_name, dst_dir):
        src_file_name = os.path.join('glance_store/tests/etc', file_name)
        shutil.copy(src_file_name, dst_dir)
        dst_file_name = os.path.join(dst_dir, file_name)
        return dst_file_name

    def config(self, **kw):
        """Override some configuration values.

        The keyword arguments are the names of configuration options to
        override and their values.

        If a group argument is supplied, the overrides are applied to
        the specified configuration option group.

        All overrides are automatically cleared at the end of the current
        test by the fixtures cleanup process.
        """
        group = kw.pop('group', None)
        for k, v in kw.items():
            if group:
                self.conf.set_override(k, v, group)
            else:
                self.conf.set_override(k, v)

    def register_store_backend_schemes(self, store, store_entry, store_identifier):
        schemes = store.get_schemes()
        scheme_map = {}
        loc_cls = store.get_store_location_class()
        for scheme in schemes:
            scheme_map[scheme] = {}
            scheme_map[scheme][store_identifier] = {'store': store, 'location_class': loc_cls, 'store_entry': store_entry}
        location.register_scheme_backend_map(scheme_map)