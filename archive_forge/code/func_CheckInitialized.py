from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def CheckInitialized(self):
    """Performs non-regular expression-based validation.

    The following are verified:
        - At least one URL mapping is provided in the URL mappers.
        - The number of URL mappers doesn't exceed `MAX_URL_MAPS`.
        - The major version does not contain the string `-dot-`.
        - If `api_endpoints` are defined, an `api_config` stanza must be
          defined.
        - If the `runtime` is `python27` and `threadsafe` is set, then no CGI
          handlers can be used.
        - The version name doesn't start with `BUILTIN_NAME_PREFIX`.
        - If `redirect_http_response_code` exists, it is in the list of valid
          300s.
        - Module and service aren't both set. Services were formerly known as
          modules.

    Raises:
      DuplicateLibrary: If `library_name` is specified more than once.
      MissingURLMapping: If no `URLMap` object is present in the object.
      TooManyURLMappings: If there are too many `URLMap` entries.
      MissingApiConfig: If `api_endpoints` exists without an `api_config`.
      MissingThreadsafe: If `threadsafe` is not set but the runtime requires it.
      ThreadsafeWithCgiHandler: If the `runtime` is `python27`, `threadsafe` is
          set and CGI handlers are specified.
      TooManyScalingSettingsError: If more than one scaling settings block is
          present.
      RuntimeDoesNotSupportLibraries: If the libraries clause is used for a
          runtime that does not support it, such as `python25`.
    """
    super(AppInfoExternal, self).CheckInitialized()
    if self.runtime is None and (not self.IsVm()):
        raise appinfo_errors.MissingRuntimeError('You must specify a "runtime" field for non-vm applications.')
    elif self.runtime is None:
        self.runtime = 'custom'
    if self.handlers and len(self.handlers) > MAX_URL_MAPS:
        raise appinfo_errors.TooManyURLMappings('Found more than %d URLMap entries in application configuration' % MAX_URL_MAPS)
    vm_runtime_python27 = self.runtime == 'vm' and (hasattr(self, 'vm_settings') and self.vm_settings and (self.vm_settings.get('vm_runtime') == 'python27')) or (hasattr(self, 'beta_settings') and self.beta_settings and (self.beta_settings.get('vm_runtime') == 'python27'))
    if self.threadsafe is None and (self.runtime == 'python27' or vm_runtime_python27):
        raise appinfo_errors.MissingThreadsafe('threadsafe must be present and set to a true or false YAML value')
    if self.auto_id_policy == DATASTORE_ID_POLICY_LEGACY:
        datastore_auto_ids_url = 'http://developers.google.com/appengine/docs/python/datastore/entities#Kinds_and_Identifiers'
        appcfg_auto_ids_url = 'http://developers.google.com/appengine/docs/python/config/appconfig#auto_id_policy'
        logging.warning("You have set the datastore auto_id_policy to 'legacy'. It is recommended that you select 'default' instead.\nLegacy auto ids are deprecated. You can continue to allocate\nlegacy ids manually using the allocate_ids() API functions.\nFor more information see:\n" + datastore_auto_ids_url + '\n' + appcfg_auto_ids_url + '\n')
    if hasattr(self, 'beta_settings') and self.beta_settings and self.beta_settings.get('source_reference'):
        ValidateCombinedSourceReferencesString(self.beta_settings.get('source_reference'))
    if self.libraries:
        if not (vm_runtime_python27 or self.runtime == 'python27'):
            raise appinfo_errors.RuntimeDoesNotSupportLibraries('libraries entries are only supported by the "python27" runtime')
        library_names = [library.name for library in self.libraries]
        for library_name in library_names:
            if library_names.count(library_name) > 1:
                raise appinfo_errors.DuplicateLibrary('Duplicate library entry for %s' % library_name)
    if self.version and self.version.find(ALTERNATE_HOSTNAME_SEPARATOR) != -1:
        raise validation.ValidationError('Version "%s" cannot contain the string "%s"' % (self.version, ALTERNATE_HOSTNAME_SEPARATOR))
    if self.version and self.version.startswith(BUILTIN_NAME_PREFIX):
        raise validation.ValidationError('Version "%s" cannot start with "%s" because it is a reserved version name prefix.' % (self.version, BUILTIN_NAME_PREFIX))
    if self.handlers:
        api_endpoints = [handler.url for handler in self.handlers if handler.GetHandlerType() == HANDLER_API_ENDPOINT]
        if api_endpoints and (not self.api_config):
            raise appinfo_errors.MissingApiConfig('An api_endpoint handler was specified, but the required api_config stanza was not configured.')
        if self.threadsafe and self.runtime == 'python27':
            for handler in self.handlers:
                if handler.script and (handler.script.endswith('.py') or '/' in handler.script):
                    raise appinfo_errors.ThreadsafeWithCgiHandler('threadsafe cannot be enabled with CGI handler: %s' % handler.script)
    if sum([bool(self.automatic_scaling), bool(self.manual_scaling), bool(self.basic_scaling)]) > 1:
        raise appinfo_errors.TooManyScalingSettingsError("There may be only one of 'automatic_scaling', 'manual_scaling', or 'basic_scaling'.")