from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import appinfo_includes
from googlecloudsdk.third_party.appengine.api import croninfo
from googlecloudsdk.third_party.appengine.api import dispatchinfo
from googlecloudsdk.third_party.appengine.api import queueinfo
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.datastore import datastore_index
class ServiceYamlInfo(_YamlInfo):
    """A class for holding some basic attributes of a parsed service yaml file."""
    DEFAULT_SERVICE_NAME = 'default'

    def __init__(self, file_path, parsed):
        """Creates a new ServiceYamlInfo.

    Args:
      file_path: str, The full path the file that was parsed.
      parsed: appinfo.AppInfoExternal, parsed Application Configuration.
    """
        super(ServiceYamlInfo, self).__init__(file_path, parsed)
        self.module = parsed.service or ServiceYamlInfo.DEFAULT_SERVICE_NAME
        if parsed.env in ['2', 'flex', 'flexible']:
            self.env = env.FLEX
        elif parsed.vm or parsed.runtime == 'vm':
            self.env = env.MANAGED_VMS
        else:
            self.env = env.STANDARD
        if self.env is env.FLEX:
            self.is_hermetic = True
        elif parsed.vm:
            for urlmap in parsed.handlers:
                if urlmap.static_dir or urlmap.static_files:
                    self.is_hermetic = False
                    break
            else:
                self.is_hermetic = True
        else:
            self.is_hermetic = False
        self._InitializeHasExplicitSkipFiles(file_path, parsed)
        self._UpdateSkipFiles(parsed)
        if self.env is env.MANAGED_VMS or self.is_hermetic:
            self.runtime = parsed.GetEffectiveRuntime()
            self._UpdateVMSettings()
        else:
            self.runtime = parsed.runtime
        self.is_ti_runtime = env.GetTiRuntimeRegistry().Get(self.runtime, self.env)

    @staticmethod
    def FromFile(file_path):
        """Parses the given service file.

    Args:
      file_path: str, The full path to the service file.

    Raises:
      YamlParseError: If the file is not a valid Yaml-file.
      YamlValidationError: If validation of parsed info fails.

    Returns:
      A ServiceYamlInfo object for the parsed file.
    """
        try:
            parsed = _YamlInfo._ParseYaml(file_path, appinfo_includes.Parse)
        except (yaml_errors.Error, appinfo_errors.Error) as e:
            raise YamlParseError(file_path, e)
        info = ServiceYamlInfo(file_path, parsed)
        info.Validate()
        return info

    def Validate(self):
        """Displays warnings and raises exceptions for non-schema errors.

    Raises:
      YamlValidationError: If validation of parsed info fails.
    """
        if self.parsed.runtime == 'vm':
            vm_runtime = self.parsed.GetEffectiveRuntime()
        else:
            vm_runtime = None
            if self.parsed.runtime == 'python':
                raise YamlValidationError('Service [{service}] uses unsupported Python 2.5 runtime. Please use [runtime: python27] instead.'.format(service=self.parsed.service or ServiceYamlInfo.DEFAULT_SERVICE_NAME))
            elif self.parsed.runtime == 'python-compat':
                raise YamlValidationError('"python-compat" is not a supported runtime.')
            elif self.parsed.runtime == 'custom' and (not self.parsed.env):
                raise YamlValidationError('runtime "custom" requires that env be explicitly specified.')
        if self.env is env.MANAGED_VMS:
            log.warning(MANAGED_VMS_DEPRECATION_WARNING)
        if self.env is env.FLEX and self.parsed.beta_settings and self.parsed.beta_settings.get('enable_app_engine_apis'):
            log.warning(APP_ENGINE_APIS_DEPRECATION_WARNING)
        if self.env is env.FLEX and vm_runtime == 'python27':
            raise YamlValidationError('The "python27" is not a valid runtime in env: flex.  Please use [python] instead.')
        if self.env is env.FLEX and vm_runtime == 'python-compat':
            log.warning('[runtime: {}] is deprecated.  Please use [runtime: python] instead.  See {} for more info.'.format(vm_runtime, UPGRADE_FLEX_PYTHON_URL))
        for warn_text in self.parsed.GetWarnings():
            log.warning('In file [{0}]: {1}'.format(self.file, warn_text))
        if self.env is env.STANDARD and self.parsed.runtime == 'python27' and HasLib(self.parsed, 'ssl', '2.7'):
            log.warning(PYTHON_SSL_WARNING)
        if self.env is env.FLEX and vm_runtime == 'python' and (GetRuntimeConfigAttr(self.parsed, 'python_version') == '3.4'):
            log.warning(FLEX_PY34_WARNING)
        _CheckIllegalAttribute(name='application', yaml_info=self.parsed, extractor_func=lambda yaml: yaml.application, file_path=self.file, msg=HINT_PROJECT)
        _CheckIllegalAttribute(name='version', yaml_info=self.parsed, extractor_func=lambda yaml: yaml.version, file_path=self.file, msg=HINT_VERSION)
        self._ValidateTi()

    def _ValidateTi(self):
        """Validation specifically for Ti-runtimes."""
        if not self.is_ti_runtime:
            return
        _CheckIllegalAttribute(name='threadsafe', yaml_info=self.parsed, extractor_func=lambda yaml: yaml.threadsafe, file_path=self.file, msg=HINT_THREADSAFE.format(self.runtime))
        for handler in self.parsed.handlers:
            _CheckIllegalAttribute(name='application_readable', yaml_info=handler, extractor_func=lambda yaml: handler.application_readable, file_path=self.file, msg=HINT_READABLE.format(self.runtime))

    def RequiresImage(self):
        """Returns True if we'll need to build a docker image."""
        return self.env is env.MANAGED_VMS or self.is_hermetic

    def _UpdateVMSettings(self):
        """Overwrites vm_settings for App Engine services with VMs.

    Also sets module_yaml_path which is needed for some runtimes.

    Raises:
      AppConfigError: if the function was called for a standard service
    """
        if self.env not in [env.MANAGED_VMS, env.FLEX]:
            raise AppConfigError('This is not an App Engine Flexible service. Please set `env` field to `flex`.')
        if not self.parsed.vm_settings:
            self.parsed.vm_settings = appinfo.VmSettings()
        self.parsed.vm_settings['module_yaml_path'] = os.path.basename(self.file)

    def GetAppYamlBasename(self):
        """Returns the basename for the app.yaml file for this service."""
        return os.path.basename(self.file)

    def HasExplicitSkipFiles(self):
        """Returns whether user has explicitly defined skip_files in app.yaml."""
        return self._has_explicit_skip_files

    def _InitializeHasExplicitSkipFiles(self, file_path, parsed):
        """Read app.yaml to determine whether user explicitly defined skip_files."""
        if getattr(parsed, 'skip_files', None) == appinfo.DEFAULT_SKIP_FILES:
            try:
                contents = files.ReadFileContents(file_path)
            except files.Error:
                contents = ''
            self._has_explicit_skip_files = 'skip_files' in contents
        else:
            self._has_explicit_skip_files = True

    def _UpdateSkipFiles(self, parsed):
        """Resets skip_files field to Flex default if applicable."""
        if self.RequiresImage() and (not self.HasExplicitSkipFiles()):
            parsed.skip_files = validation._RegexStrValue(validation.Regex(DEFAULT_SKIP_FILES_FLEX), DEFAULT_SKIP_FILES_FLEX, 'skip_files')