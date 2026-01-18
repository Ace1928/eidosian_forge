import os
import requests
from keystoneauth1 import loading
from keystoneauth1 import plugin
from oslo_log import log
class VitrageNoAuthLoader(loading.BaseLoader):
    plugin_class = VitrageNoAuthPlugin

    def get_options(self):
        options = super(VitrageNoAuthLoader, self).get_options()
        options.extend([VitrageOpt('user-id', help='User ID', required=True), VitrageOpt('project-id', help='Project ID', required=True), VitrageOpt('roles', help='Roles', default='admin'), VitrageOpt('endpoint', help='Vitrage endpoint', required=True)])
        return options