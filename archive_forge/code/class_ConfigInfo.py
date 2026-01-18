from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import getpass
import io
import locale
import os
import platform as system_platform
import re
import ssl
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import http_proxy_setup
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import requests
import six
import urllib3
class ConfigInfo(object):
    """Holds information about where config is stored and what values are set."""

    def __init__(self, anonymizer=None):
        anonymizer = anonymizer or NoopAnonymizer()
        cfg_paths = config.Paths()
        active_config = named_configs.ConfigurationStore.ActiveConfig()
        self.active_config_name = active_config.name
        self.paths = {'installation_properties_path': anonymizer.ProcessPath(cfg_paths.installation_properties_path), 'global_config_dir': anonymizer.ProcessPath(cfg_paths.global_config_dir), 'active_config_path': anonymizer.ProcessPath(active_config.file_path), 'sdk_root': anonymizer.ProcessPath(cfg_paths.sdk_root)}
        self.account = anonymizer.ProcessAccount(properties.VALUES.core.account.Get(validate=False))
        self.project = anonymizer.ProcessProject(properties.VALUES.core.project.Get(validate=False))
        self.universe_domain = anonymizer.ProcessUniverseDomain(properties.VALUES.core.universe_domain.Get(validate=False))
        self.properties = properties.VALUES.AllPropertyValues()
        if self.properties.get('core', {}).get('account'):
            self.properties['core']['account'].value = anonymizer.ProcessAccount(self.properties['core']['account'].value)
        if self.properties.get('core', {}).get('project'):
            self.properties['core']['project'].value = anonymizer.ProcessProject(self.properties['core']['project'].value)
        if self.properties.get('core', {}).get('universe_domain'):
            self.properties['core']['universe_domain'].value = anonymizer.ProcessUniverseDomain(self.properties['core']['universe_domain'].value)
        if self.properties.get('proxy', {}).get('username'):
            self.properties['proxy']['username'].value = anonymizer.ProcessUsername(self.properties['proxy']['username'].value)
        if self.properties.get('proxy', {}).get('password'):
            self.properties['proxy']['password'].value = anonymizer.ProcessPassword(self.properties['proxy']['password'].value)

    def __str__(self):
        out = io.StringIO()
        out.write('Installation Properties: [{0}]\n'.format(self.paths['installation_properties_path']))
        out.write('User Config Directory: [{0}]\n'.format(self.paths['global_config_dir']))
        out.write('Active Configuration Name: [{0}]\n'.format(self.active_config_name))
        out.write('Active Configuration Path: [{0}]\n\n'.format(self.paths['active_config_path']))
        out.write('Account: [{0}]\n'.format(self.account))
        out.write('Project: [{0}]\n'.format(self.project))
        all_cred_accounts = c_store.AllAccountsWithUniverseDomains()
        for account in all_cred_accounts:
            if account.account == self.account:
                cred_universe_domain = account.universe_domain
                break
        else:
            cred_universe_domain = None
        if cred_universe_domain and cred_universe_domain != self.universe_domain:
            domain_mismatch_warning = ' WARNING: Mismatch with universe domain of account'
        else:
            domain_mismatch_warning = ''
        out.write('Universe Domain: [{0}]{1}\n\n'.format(self.universe_domain, domain_mismatch_warning))
        out.write('Current Properties:\n')
        for section, props in six.iteritems(self.properties):
            out.write('  [{section}]\n'.format(section=section))
            for name, property_value in six.iteritems(props):
                out.write('    {name}: [{value}] ({source})\n'.format(name=name, value=str(property_value.value), source=property_value.source.value))
        return out.getvalue()