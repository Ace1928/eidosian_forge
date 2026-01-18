from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import hashlib
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import release_notes
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_check
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
from six.moves import map  # pylint: disable=redefined-builtin
class UpdateManager(object):
    """Main class for performing updates for the Google Cloud CLI."""
    BIN_DIR_NAME = 'bin'
    VERSIONED_SNAPSHOT_FORMAT = 'components-v{0}.json'
    DARWIN_X86_64 = platforms.Platform(platforms.OperatingSystem.MACOSX, platforms.Architecture.x86_64)

    @staticmethod
    def GetAdditionalRepositories():
        """Gets the currently registered repositories as a list.

    Returns:
      [str], The list of registered repos or [] if there are none.
    """
        repos = properties.VALUES.component_manager.additional_repositories.Get()
        if repos:
            return repos.split(',')
        return []

    @staticmethod
    def EnsureInstalledAndRestart(components, msg=None, command=None):
        """Installs the given components if necessary and then restarts gcloud.

    Args:
      components: [str], The components that must be installed.
      msg: str, A custom message to print.
      command: str, the command to run, if not `gcloud`

    Returns:
      bool, True if the components were already installed.  If installation must
      occur, this method never returns because gcloud is reinvoked after the
      update is done.

    Raises:
      MissingRequiredComponentsError: If the components are not installed and
      the user chooses not to install them.
    """
        platform = platforms.Platform.Current()
        manager = UpdateManager(platform_filter=platform, warn=False)
        return manager._EnsureInstalledAndRestart(components, msg, command)

    @staticmethod
    def UpdatesAvailable():
        """Returns True if updates are available, False otherwise.

    Returns:
      bool, True if updates are available, False otherwise.
    """
        if config.INSTALLATION_CONFIG.disable_updater or properties.VALUES.component_manager.disable_update_check.GetBool():
            log.debug('SDK update checks are disabled.')
            return False
        with update_check.UpdateCheckData() as last_update_check:
            return last_update_check.UpdatesAvailable()

    @staticmethod
    def PerformUpdateCheck(command_path, force=False):
        """Checks to see if a new snapshot has been released periodically.

    This method can be called as often as you'd like.  It will only actually
    check the server for updates if a certain amount of time has elapsed since
    the last check (or if force is True).  If updates are available, to any
    installed components, it will print a notification message.

    Args:
      command_path: str, The '.' separated path of the command that is currently
        being run (i.e. gcloud.foo.bar).
      force: bool, True to force a server check for updates, False to check only
        if the update frequency has expired.
    """
        if config.INSTALLATION_CONFIG.disable_updater or properties.VALUES.component_manager.disable_update_check.GetBool():
            log.debug('SDK update checks are disabled.')
            return
        platform = platforms.Platform.Current()
        manager = UpdateManager(platform_filter=platform, warn=False)
        manager._PerformUpdateCheck(command_path, force=force)

    def __init__(self, sdk_root=None, url=None, platform_filter=None, warn=None, skip_compile_python=False):
        """Creates a new UpdateManager.

    Args:
      sdk_root: str, The path to the root directory of the Cloud SDK is
        installation.  If None, the updater will search for the install
        directory based on the current directory.
      url: str, The URL to get the latest component snapshot from.  If None,
        the default will be used.
      platform_filter: platforms.Platform, A platform that components must match
        in order to be considered for any operations.  If None, only components
        without OS or architecture filters will match.
      warn: bool, True to warn about overridden configuration like an alternate
        snapshot file, fixed SDK version, or additional repo.  Should be set
        to False when using this class for background operations like checking
        for updates so the user only sees the warnings when they are actually
        dealing directly with the component manager.
      skip_compile_python: bool, if True the UpdateManager will not pre compile
      installed python files.

    Raises:
      local_state.InvalidSDKRootError: If the Cloud SDK root cannot be found.
      MissingUpdateURLError: If we don't know what manifest to download.
    """
        self.__sdk_root = sdk_root
        if not self.__sdk_root:
            self.__sdk_root = config.Paths().sdk_root
        if not self.__sdk_root:
            raise local_state.InvalidSDKRootError()
        self.__sdk_root = encoding.Decode(self.__sdk_root)
        if warn is None:
            warn = not properties.VALUES.component_manager.disable_warning.GetBool()
        if not url:
            url = properties.VALUES.component_manager.snapshot_url.Get()
        if url:
            if warn:
                log.warning('You are using an overridden snapshot URL: [%s]', url)
        else:
            url = properties.VALUES.component_manager.original_snapshot_url.Get()
        if not url:
            raise MissingUpdateURLError()
        self.__sdk_root = os.path.realpath(self.__sdk_root)
        self.__base_url = url
        self.__platform_filter = platform_filter
        self.__text_wrapper = textwrap.TextWrapper(replace_whitespace=False, drop_whitespace=False)
        self.__warn = warn
        self.__enable_fallback = self._EnableFallback()
        fixed_version = properties.VALUES.component_manager.fixed_sdk_version.Get()
        self.__fixed_version = fixed_version
        self.__skip_compile_python = skip_compile_python

    def _EnableFallback(self):
        return self.__platform_filter and self.__platform_filter.operating_system == platforms.OperatingSystem.MACOSX and (self.__platform_filter.architecture == platforms.Architecture.arm)

    def __Write(self, stream, msg='', word_wrap=False):
        """Writes the given message to the out stream with a new line.

    Args:
      stream:  The output stream to write to.
      msg: str, The message to write.
      word_wrap: bool, True to enable nicer word wrapper, False to just print
        the string as is.
    """
        if word_wrap:
            msg = self.__text_wrapper.fill(msg)
        stream.write(msg + '\n')
        stream.flush()

    def _ShouldDoFastUpdate(self, allow_no_backup=False, fast_mode_impossible=False, has_components_to_remove=False):
        """Determine whether we should do an in-place fast update or make a backup.

    This method also ensures the CWD is valid for the mode we are going to use.

    Args:
      allow_no_backup: bool, True if we want to allow the updater to run
        without creating a backup.  This lets us be in the root directory of the
        SDK and still do an update.  It is more fragile if there is a failure,
        so we only do it if necessary.
      fast_mode_impossible: bool, True if we can't do a fast update for this
        particular operation (overrides forced fast mode).
      has_components_to_remove: bool, Whether the update operation involves
        removing any components. Affects whether we can perform an in-place
        update from inside the root dir.

    Returns:
      bool, True if allow_no_backup was True and we are under the SDK root (so
        we should do a no backup update).

    Raises:
      InvalidCWDError: If the command is run from a directory within the SDK
        root.
    """
        force_fast = properties.VALUES.experimental.fast_component_update.GetBool()
        if fast_mode_impossible:
            force_fast = False
        cwd = None
        try:
            cwd = os.path.realpath(file_utils.GetCWD())
        except OSError:
            log.debug('Could not determine CWD, assuming detached directory not under SDK root.')
        if not (cwd and file_utils.IsDirAncestorOf(self.__sdk_root, cwd)):
            return force_fast
        if (allow_no_backup or force_fast) and (not has_components_to_remove):
            return True
        raise InvalidCWDError('Your current working directory is inside the Google Cloud CLI install root: {root}.  In order to perform this update, run the command from outside of this directory.'.format(root=self.__sdk_root))

    def _GetDontCancelMessage(self, disable_backup):
        """Get the message to print before udpates.

    Args:
      disable_backup: bool, True if we are doing an in place update.

    Returns:
      str, The message to print, or None.
    """
        if disable_backup:
            return 'Once started, canceling this operation may leave your SDK installation in an inconsistent state.'
        else:
            return None

    def _GetMappingFile(self, filename):
        """Checks if mapping files are present and loads them for further use.

    Args:
      filename: str, The full filename (with .yaml extension) to be loaded.

    Returns:
      Loaded YAML if mapping files are present, None otherwise.
    """
        paths = config.Paths()
        mapping_path = os.path.join(paths.sdk_root, paths.CLOUDSDK_STATE_DIR, filename)
        if os.path.isfile(mapping_path):
            return yaml.load_path(mapping_path)

    def _ComputeMappingMessage(self, command, commands_map, components_map, components=None):
        """Returns error message containing correct command mapping.

    Checks the user-provided command to see if it maps to one we support for
    their package manager. If it does, compute error message to let the user
    know why their command did not work and provide them with an alternate,
    accurate command to run. If we do not support the given command/component
    combination for their package manager, provide user with instructions to
    change their package manager.

    Args:
      command: str, Command from user input, to be mapped against
        commands_mapping.yaml
      commands_map: dict, Contains mappings from commands_mapping.yaml
      components_map: dict, Contains mappings from components_mapping.yaml
      components: str list, Component from user input, to be mapped against
        component_commands.yaml

    Returns:
      str, The compiled error message.
    """
        final_message = ''
        unavailable = 'unavailable'
        update_all = 'update-all'
        unavailable_components = None
        mapped_components = None
        not_components = None
        mapped_packages = None
        correct_command = commands_map[command]
        if components:
            not_components = [component for component in components if component not in components_map]
            unavailable_components = [component for component in components if components_map.get(component) == unavailable]
        if command == update_all:
            mapped_packages = [component for component in set(components_map.values()) if component != unavailable]
        else:
            mapped_components = [component for component in components if component not in unavailable_components and component not in not_components]
            mapped_packages = [components_map[component] for component in mapped_components]
        if mapped_packages:
            correct_command = correct_command.format(package=' '.join(mapped_packages))
        if mapped_packages or not components:
            final_message += '\nYou cannot perform this action because the Google Cloud CLI component manager \nis disabled for this installation. You can run the following command \nto achieve the same result for this installation: \n\n{correct_command}\n\n'.format(correct_command=correct_command)
        if unavailable_components:
            final_message += '\nThe {component} component(s) is unavailable through the packaging system \nyou are currently using. Please consider using a separate installation \nof the Google Cloud CLI created through the default mechanism described at: \n\n{doc_url} \n\n'.format(component=', '.join(unavailable_components), doc_url=config.INSTALLATION_CONFIG.documentation_url)
        if not_components:
            final_message += '"{component}" are not valid component name(s).\n'.format(component=', '.join(not_components))
        return final_message

    def _CheckIfDisabledAndThrowError(self, components=None, command=None):
        """Checks if updater is disabled. If so, raises UpdaterDisabledError.

    The updater is disabled for installations that come from other package
    managers like apt-get or if the current user does not have permission
    to create or delete files in the SDK root directory. If disabled, raises
    UpdaterDisabledError either with the default message, or an error message
    from _ComputeMappingMessage if a command was passed in.

    Args:
      components: str list, Component from user input, to be mapped against
        component_commands.yaml
      command: str, Command from user input, to be mapped against
        command_mapping.yaml

    Raises:
      UpdaterDisabledError: If the updater is disabled.
    """
        default_message = 'You cannot perform this action because this Google Cloud CLI installation is managed by an external package manager.\nPlease consider using a separate installation of the Google Cloud CLI created through the default mechanism described at: {doc_url}\n'.format(doc_url=config.INSTALLATION_CONFIG.documentation_url)
        if config.INSTALLATION_CONFIG.disable_updater:
            if not command:
                raise UpdaterDisabledError(default_message)
            commands_map = self._GetMappingFile(filename='command_mapping.yaml')
            components_map = self._GetMappingFile(filename='component_mapping.yaml')
            if not (components_map and commands_map):
                raise UpdaterDisabledError(default_message)
            mapping_message = self._ComputeMappingMessage(command, commands_map, components_map, components)
            raise UpdaterDisabledError(mapping_message)

    def _GetInstallState(self):
        return local_state.InstallationState(self.__sdk_root)

    def _GetEffectiveSnapshotURL(self, version=None):
        """Get the snapshot URL we shoould download based on any override versions.

    This starts with the configured URL (or comma separated URL list) and
    potentially modifies it based on the version.  If a version is specified,
    it is converted to the fixed version specific snapshot.  If the SDK is set
    to use a fixed version, that is then used.  If neither, the original URL
    is used.

    Args:
      version: str, The Cloud SDK version to get the snapshot for.

    Raises:
      MismatchedFixedVersionsError: If you manually specify a version and you
        are fixed to a different version.

    Returns:
      str, The modified snapshot URL.
    """
        url = self.__base_url
        if version:
            if self.__fixed_version and self.__fixed_version != version:
                raise MismatchedFixedVersionsError('You have configured your Google Cloud CLI installation\nto be fixed to version [{0}] but are attempting to install components at\nversion [{1}].  To clear your fixed version setting, run:\n    $ gcloud config unset component_manager/fixed_sdk_version'.format(self.__fixed_version, version))
        elif self.__fixed_version:
            if self.__warn:
                log.warning('You have configured your Google Cloud CLI installation to be fixed to version [{0}].'.format(self.__fixed_version))
            version = self.__fixed_version
        if version:
            urls = url.split(',')
            urls[0] = os.path.dirname(urls[0]) + '/' + UpdateManager.VERSIONED_SNAPSHOT_FORMAT.format(version)
            url = ','.join(urls)
        repos = properties.VALUES.component_manager.additional_repositories.Get()
        if repos:
            if self.__warn:
                for repo in repos.split(','):
                    log.warning('You are using additional component repository: [%s]', repo)
            url = ','.join([url, repos])
        return url

    def _GetLatestSnapshot(self, version=None, command_path='unknown'):
        effective_url = self._GetEffectiveSnapshotURL(version=version)
        try:
            return snapshots.ComponentSnapshot.FromURLs(*effective_url.split(','), command_path=command_path)
        except snapshots.URLFetchError:
            if version:
                log.error('The component listing for Google Cloud CLI version [{0}] could not be found.  Make sure this is a valid archived Google Cloud CLI version.'.format(version))
            elif self.__fixed_version:
                log.error('You have configured your Google Cloud CLI installation to be fixed to version [{0}]. Make sure this is a valid archived Google Cloud CLI version.'.format(self.__fixed_version))
            raise

    def _GetStateAndDiff(self, version=None, command_path='unknown'):
        install_state = self._GetInstallState()
        latest_snapshot = self._GetLatestSnapshot(version=version, command_path=command_path)
        diff = install_state.DiffCurrentState(latest_snapshot, platform_filter=self.__platform_filter)
        return (install_state, diff)

    def GetCurrentVersionsInformation(self, include_hidden=False):
        """Get the current version for every installed component.

    Args:
      include_hidden: bool, include hidden components.

    Returns:
      {str:str}, A mapping from component id to version string.
    """
        current_state = self._GetInstallState()
        versions = {}
        installed_components = current_state.InstalledComponents()
        for component_id, component in six.iteritems(installed_components):
            component_def = component.ComponentDefinition()
            if component_def.is_configuration or (component_def.is_hidden and (not include_hidden)):
                continue
            versions[component_id] = _GetVersionString(component, installed_components)
        return versions

    def _PerformUpdateCheck(self, command_path, force=False):
        """Checks to see if a new snapshot has been released periodically.

    This method can be called as often as you'd like.  It will only actually
    check the server for updates if a certain amount of time has elapsed since
    the last check (or if force is True).  If updates are available, to any
    installed components, it will print a notification message.

    Args:
      command_path: str, The '.' separated path of the command that is currently
        being run (i.e. gcloud.foo.bar).
      force: bool, True to force a server check for updates, False to check only
        if the update frequency has expired.
    """
        with update_check.UpdateCheckData() as last_update_check:
            if force or last_update_check.ShouldDoUpdateCheck():
                log.debug('Checking for updates...')
                try:
                    _, diff = self._GetStateAndDiff(command_path=installers.UPDATE_MANAGER_COMMAND_PATH)
                    last_update_check.SetFromSnapshot(diff.latest, bool(diff.AvailableUpdates()), force=force)
                except snapshots.IncompatibleSchemaVersionError:
                    last_update_check.SetFromIncompatibleSchema()
            last_update_check.Notify(command_path)

    def List(self, show_hidden=False, only_local_state=False):
        """Lists all of the components and their current state.

    This pretty prints the list of components along with whether they are up
    to date, require an update, etc.

    Args:
      show_hidden: bool, include hidden components.
      only_local_state: bool, only return component information for local state.

    Returns:
      The list of snapshots.ComponentDiffs (or snapshots.ComponentInfos if
      only_local_state is True) for all components that are not hidden.
    """
        if only_local_state:
            to_print, current_version = self._GetPrintListOnlyLocal()
            latest_version = None
        else:
            to_print, current_version, latest_version = self._GetPrintListWithDiff()
        if not show_hidden:
            to_print = [c for c in to_print if not c.is_hidden]
        return (to_print, current_version, latest_version)

    def _GetPrintListOnlyLocal(self):
        """Helper method that gets a list of locally installed components to print.

    Returns:
      List of snapshots.ComponentInfos for the List method as well as the
      current version string.
    """
        install_state = self._GetInstallState()
        to_print = install_state.Snapshot().CreateComponentInfos(platform_filter=self.__platform_filter)
        if self._EnableFallback():
            native_ids = set((c.id for c in to_print))
            darwin_x86_64_all = install_state.Snapshot().CreateComponentInfos(platform_filter=self.DARWIN_X86_64)
            to_print_x86_64 = (c for c in darwin_x86_64_all if c.id not in native_ids)
            to_print.extend(to_print_x86_64)
        current_version = config.INSTALLATION_CONFIG.version
        self.__Write(log.status, '\nYour current Google Cloud CLI version is: ' + current_version)
        return (to_print, current_version)

    def _GetPrintListWithDiff(self):
        """Helper method that computes a diff and returns a list of diffs to print.

    Returns:
      List of snapshots.ComponentDiffs for the List method as well as the
      current and latest version strings.
    """
        try:
            _, diff = self._GetStateAndDiff(command_path='components.list')
        except snapshots.IncompatibleSchemaVersionError as e:
            self._ReinstallOnError(e)
            return ([], None, None)
        latest_msg = 'The latest available version is: ' if not self.__fixed_version else None
        current_version, latest_version = self._PrintVersions(diff, latest_msg=latest_msg)
        to_print = diff.AvailableUpdates() + diff.Removed() + diff.AvailableToInstall() + diff.UpToDate()
        return (to_print, current_version, latest_version)

    def _PrintVersions(self, diff, latest_msg=None):
        """Prints the current and latest version.

    Args:
      diff: snapshots.ComponentSnapshotDiff, The snapshot diff we are working
        with.
      latest_msg: str, The message to print when displaying the latest version.
        If None, nothing about the latest version is printed.

    Returns:
      (str, str), The current and latest version strings.
    """
        current_version = config.INSTALLATION_CONFIG.version
        latest_version = diff.latest.version
        self.__Write(log.status, '\nYour current Google Cloud CLI version is: ' + current_version)
        if latest_version and latest_msg:
            self.__Write(log.status, latest_msg + latest_version)
        self.__Write(log.status)
        return (current_version, latest_version)

    def _HashRcfiles(self, shell_rc_files):
        """Creates the sha256 checksums of files.

    Args:
      shell_rc_files: list, A list of files to get the sha256 checksums.
    Returns:
      sha256dict, dictionary of sha256 file sums.
    """
        sha256dict = {}
        for name in shell_rc_files:
            try:
                fpath = os.path.join(self.__sdk_root, name)
                if not os.path.exists(fpath):
                    continue
                sha256 = hashlib.sha256(file_utils.ReadBinaryFileContents(fpath)).hexdigest()
                sha256dict[name] = sha256
            except OSError:
                sha256dict[name] = 0
                continue
        return sha256dict

    def _PrintPendingAction(self, components, action):
        """Prints info about components we are going to install or remove.

    Args:
      components: list(schemas.Component), The components that are going to be
        acted on.
      action: str, The verb to print for this set of components.
    """
        attributes = ['box', 'title="These components will be {action}."'.format(action=action)]
        columns = ['details.display_name:label=Name:align=left', 'version.version_string:label=Version:align=right', 'data.size.size(zero="",min=1048576):label=Size:align=right']
        fmt = 'table[{attributes}]({columns})'.format(attributes=','.join(attributes), columns=','.join(columns))
        resource_printer.Print(components, fmt, out=log.status)

    def _UpdateWithProgressBar(self, components, action, action_func, first=False, last=False):
        """Performs an update on a component while using a progress bar.

    Args:
      components: [schemas.Component], The components that are going to be acted
        on.
      action: str, The action that is printed for this update.
      action_func: func, The function to call to actually do the update.  It
        takes a single argument which is the component id.
      first: bool, True if this is the first stacked ProgressBar group.
      last: bool, True if this is the last stacked ProgressBar group.
    """
        for index, component in enumerate(components):
            label = '{action}: {name}'.format(action=action, name=component.details.display_name)
            with console_io.ProgressBar(label=label, stream=log.status, first=first, last=last and index == len(components) - 1) as pb:
                action_func(component.id, progress_callback=pb.SetProgress)
            first = False

    def _InstallFunction(self, install_state, diff):

        def Inner(component_id, progress_callback):
            return install_state.Install(diff.latest, component_id, progress_callback=progress_callback, command_path='components.update')
        return Inner

    def Install(self, components, allow_no_backup=False, throw_if_unattended=False, restart_args=None):
        """Installs the given components at the version you are current on.

    Args:
      components: [str], A list of component ids to install.
      allow_no_backup: bool, True if we want to allow the updater to run
        without creating a backup.  This lets us be in the root directory of the
        SDK and still do an update.  It is more fragile if there is a failure,
        so we only do it if necessary.
      throw_if_unattended: bool, True to throw an exception on prompts when
        not running in interactive mode.
      restart_args: list of str or None. If given, this gcloud command should be
        run in event of a restart (ex. if we're using a bundled Python
        installation to do this update).

    Raises:
      InvalidComponentError: If any of the given component ids do not exist.

    Returns:
      bool, True if the update succeeded (or there was nothing to do, False if
      if was cancelled by the user.
    """
        if not components:
            raise InvalidComponentError('You must specify components to install')
        version = config.INSTALLATION_CONFIG.version
        if properties.VALUES.component_manager.additional_repositories.Get():
            log.warning('Additional component repositories are currently active.  Running `update` instead of `install`.')
            version = None
        return self.Update(components, allow_no_backup=allow_no_backup, throw_if_unattended=throw_if_unattended, version=version, restart_args=restart_args)

    def Update(self, update_seed=None, allow_no_backup=False, throw_if_unattended=False, version=None, restart_args=None):
        """Performs an update of the given components.

    If no components are provided, it will attempt to update everything you have
    installed.

    Args:
      update_seed: list of str, A list of component ids to update.
      allow_no_backup: bool, True if we want to allow the updater to run
        without creating a backup.  This lets us be in the root directory of the
        SDK and still do an update.  It is more fragile if there is a failure,
        so we only do it if necessary.
      throw_if_unattended: bool, True to throw an exception on prompts when
        not running in interactive mode.
      version: str, The SDK version to update to instead of latest.
      restart_args: list of str or None. If given, this gcloud command should be
        run in event of a restart (ex. if we're using a bundled Python
        installation to do this update).

    Returns:
      bool, True if the update succeeded (or there was nothing to do, False if
      if was cancelled by the user.

    Raises:
      InvalidComponentError: If any of the given component ids do not exist.
    """
        sha256dict1 = self._HashRcfiles(_SHELL_RCFILES)
        if update_seed:
            self._CheckIfDisabledAndThrowError(components=update_seed, command='update')
        else:
            self._CheckIfDisabledAndThrowError(command='update-all')
        try:
            install_state, diff = self._GetStateAndDiff(version=version, command_path='components.update')
        except snapshots.IncompatibleSchemaVersionError as e:
            return self._ReinstallOnError(e)
        original_update_seed = update_seed
        if update_seed:
            update_seed = self._HandleInvalidUpdateSeeds(diff, version, update_seed)
        else:
            update_seed = list(diff.current.components.keys())
        to_remove = diff.ToRemove(update_seed)
        to_install = diff.ToInstall(update_seed)
        self._HandleBadComponentInstallState(update_seed, to_install)
        self.__Write(log.status)
        if not to_remove and (not to_install):
            self.__Write(log.status, 'All components are up to date.')
            with update_check.UpdateCheckData() as last_update_check:
                last_update_check.SetFromSnapshot(diff.latest, bool(diff.AvailableUpdates()), force=True)
            install_state.ClearDeprecatedDirs()
            return True
        config.EnsureSDKWriteAccess(self.__sdk_root)
        if self.IsPythonBundled() and _ContainsBundledPython(to_remove):
            _HaltIfBundledPythonUnix()
            log.warning(BUNDLED_PYTHON_REMOVAL_WARNING)
        self._RestartIfUsingBundledPython(args=restart_args)
        if original_update_seed:
            latest_msg = 'Installing components from version: '
        else:
            latest_msg = 'You will be upgraded to version: '
        current_version, _ = self._PrintVersions(diff, latest_msg=latest_msg)
        disable_backup = self._ShouldDoFastUpdate(allow_no_backup=allow_no_backup, has_components_to_remove=bool(to_remove))
        self._PrintPendingAction(FilterMetaComponents(diff.DetailsForCurrent(to_remove - to_install)), 'removed')
        self._PrintPendingAction(FilterMetaComponents(diff.DetailsForLatest(to_remove & to_install)), 'updated')
        self._PrintPendingAction(FilterMetaComponents(diff.DetailsForLatest(to_install - to_remove)), 'installed')
        self.__Write(log.status)
        release_notes.PrintReleaseNotesDiff(diff.latest.sdk_definition.release_notes_url, config.INSTALLATION_CONFIG.version, diff.latest.version)
        message = self._GetDontCancelMessage(disable_backup)
        if not console_io.PromptContinue(message=message, throw_if_unattended=throw_if_unattended):
            return False
        components_to_install = diff.DetailsForLatest(to_install)
        components_to_remove = diff.DetailsForCurrent(to_remove)
        for c in components_to_install:
            metrics.Installs(c.id, c.version.version_string)
        if disable_backup:
            with execution_utils.UninterruptibleSection(stream=log.status):
                self.__Write(log.status, 'Performing in place update...\n')
                with self._EnsureCaCertsCleanedUpIfObsoleted() as ca_certs_path:
                    with _EnsureCaCertsRestoredIfDeleted(ca_certs_path):
                        self._UpdateWithProgressBar(components_to_remove, 'Uninstalling', install_state.Uninstall, first=True, last=not components_to_install)
                    self._UpdateWithProgressBar(components_to_install, 'Installing', self._InstallFunction(install_state, diff), first=not components_to_remove, last=True)
        else:
            with console_io.ProgressBar(label='Creating update staging area', stream=log.status, last=False) as pb:
                staging_state = install_state.CloneToStaging(pb.SetProgress)
            self._UpdateWithProgressBar(components_to_remove, 'Uninstalling', staging_state.Uninstall, first=False, last=False)
            self._UpdateWithProgressBar(components_to_install, 'Installing', self._InstallFunction(staging_state, diff), first=False, last=False)
            with console_io.ProgressBar(label='Creating backup and activating new installation', stream=log.status, first=False) as pb:
                install_state.ReplaceWith(staging_state, pb.SetProgress)
        install_state.ClearDeprecatedDirs()
        with update_check.UpdateCheckData() as last_update_check:
            new_diff = install_state.DiffCurrentState(diff.latest, platform_filter=self.__platform_filter)
            last_update_check.SetFromSnapshot(new_diff.latest, bool(new_diff.AvailableUpdates()), force=True)
        self._PostProcess(snapshot=diff.latest)
        sha256dict2 = self._HashRcfiles(_SHELL_RCFILES)
        if sha256dict1 != sha256dict2:
            self.__Write(log.status, console_io.FormatRequiredUserAction('Start a new shell for the changes to take effect.'))
        self.__Write(log.status, '\nUpdate done!\n')
        if not original_update_seed:
            self.__Write(log.status, 'To revert your CLI to the previously installed version, you may run:\n  $ gcloud components update --version {current}\n'.format(current=current_version), word_wrap=False)
        if self.__warn and (not encoding.GetEncodedValue(os.environ, 'CLOUDSDK_REINSTALL_COMPONENTS')):
            bad_commands = self.FindAllOtherToolsOnPath()
            if bad_commands:
                log.warning('  There are other instances of Google Cloud tools on your system PATH.\n  Please remove the following to avoid confusion or accidental invocation:\n\n  {0}\n\n  '.format('\n'.join(bad_commands)))
            duplicate_commands = self.FindAllDuplicateToolsOnPath()
            if duplicate_commands:
                log.warning('  There are alternate versions of the following Google Cloud tools on\n  your system PATH. Please double check your PATH:\n\n  {0}\n\n  '.format('\n  '.join(duplicate_commands)))
        return True

    @contextlib.contextmanager
    def _EnsureCaCertsCleanedUpIfObsoleted(self):
        """Context manager that cleans up CA certs file upon exit if it's obsolete.

    When performing an in-place update, we restore the CA certs file to its
    original location if the component containing it needed to be removed. It's
    possible, however, that a newer version of gcloud gets the CA certs file
    from a different path (e.g. if there's a change to certifi or gcloud's
    directory structure at some point in the future). In that case, we want to
    ensure we clean up the old path once it's no longer needed to make requests
    (i.e. just after installing the new components), rather than have it lying
    around forever in the install dir. This context manager takes care of that.

    Yields:
      str, Path to the existing CA certs file.
    """
        ca_certs_path = properties.VALUES.core.custom_ca_certs_file.Get() or certifi.where()
        initial_state = self._GetInstallState()
        initial_components = initial_state.InstalledComponents()
        initial_paths = set(itertools.chain.from_iterable((manifest.InstalledPaths() for manifest in initial_components.values())))
        try:
            yield ca_certs_path
        finally:
            current_state = self._GetInstallState()
            current_components = current_state.InstalledComponents()
            current_paths = set(itertools.chain.from_iterable((manifest.InstalledPaths() for manifest in current_components.values())))
            removed_paths = initial_paths - current_paths
            try:
                relative_ca_certs_path = pathlib.Path(ca_certs_path).resolve().relative_to(pathlib.Path(self.__sdk_root).resolve()).as_posix()
            except ValueError:
                pass
            else:
                if relative_ca_certs_path in removed_paths:
                    os.remove(ca_certs_path)
                    dir_path = os.path.dirname(os.path.normpath(relative_ca_certs_path))
                    full_dir_path = os.path.join(self.__sdk_root, dir_path)
                    while dir_path and (not os.listdir(full_dir_path)):
                        os.rmdir(full_dir_path)
                        dir_path = os.path.dirname(dir_path)
                        full_dir_path = os.path.join(self.__sdk_root, dir_path)

    def _HandleBadComponentInstallState(self, update_seed, to_install):
        """Deal with bad install state of kubectl component on darwin-arm machines.

    Args:
      update_seed: list of str, A list of component ids to update.
      to_install: set of str, A set of component ids to install.

    """
        if self._EnableFallback() and 'kubectl' in set(update_seed) | to_install:
            if not os.path.isfile(self.__sdk_root + '/bin/kubectl'):
                to_install.add('kubectl-darwin-arm')

    def _HandleInvalidUpdateSeeds(self, diff, version, update_seed):
        """Checks that the update seeds are valid components.

    Args:
      diff: The ComponentSnapshotDiff.
      version: str, The SDK version if in install mode or None if in update
        mode.
      update_seed: [str], A list of component ids to update.

    Raises:
      InvalidComponentError: If any of the given component ids do not exist.

    Returns:
      [str], The update seeds that should be used for the install/update.
    """
        invalid_seeds = diff.InvalidUpdateSeeds(update_seed)
        if not invalid_seeds:
            return update_seed
        if encoding.GetEncodedValue(os.environ, 'CLOUDSDK_REINSTALL_COMPONENTS'):
            return set(update_seed) - invalid_seeds
        ignored = set(_IGNORED_MISSING_COMPONENTS)
        deprecated = invalid_seeds & ignored
        for item in deprecated:
            log.warning('Component [%s] no longer exists.', item)
            additional_msg = _IGNORED_MISSING_COMPONENTS.get(item)
            if additional_msg:
                log.warning(additional_msg)
        invalid_seeds -= ignored
        if invalid_seeds:
            completely_invalid_seeds = invalid_seeds
            update_required_seeds = set()
            if version:
                _, latest_diff = self._GetStateAndDiff(command_path='components.update')
                completely_invalid_seeds = latest_diff.InvalidUpdateSeeds(invalid_seeds)
                update_required_seeds = invalid_seeds - completely_invalid_seeds
            msgs = []
            if completely_invalid_seeds:
                msgs.append('The following components are unknown [{}].'.format(', '.join(completely_invalid_seeds)))
            if update_required_seeds:
                msgs.append('The following components are not available for your current CLI version [{}]. Please run `gcloud components update` to update your Google Cloud CLI.'.format(', '.join(update_required_seeds)))
            raise InvalidComponentError(' '.join(msgs))
        return set(update_seed) - deprecated

    def _FindToolsOnPath(self, path=None, duplicates=True, other=True):
        """Helper function to find commands matching SDK bin dir on the path."""
        bin_dir = os.path.realpath(os.path.join(self.__sdk_root, UpdateManager.BIN_DIR_NAME))
        if not os.path.exists(bin_dir):
            return set()
        commands = [f for f in os.listdir(bin_dir) if os.path.isfile(os.path.join(bin_dir, f)) and (not f.startswith('.'))]
        duplicates_in_sdk_root = set()
        bad_commands = set()
        for command in commands:
            existing_paths = file_utils.SearchForExecutableOnPath(command, path=path)
            if existing_paths:
                this_tool = os.path.join(bin_dir, command)
                if other:
                    bad_commands.update(set((os.path.realpath(f) for f in existing_paths if self.__sdk_root not in os.path.realpath(f))) - set([this_tool]))
                if duplicates:
                    duplicates_in_sdk_root.update(set((os.path.realpath(f) for f in existing_paths if self.__sdk_root in os.path.realpath(f))) - set([this_tool]))
        return bad_commands.union(duplicates_in_sdk_root)

    def FindAllOtherToolsOnPath(self, path=None):
        """Searches the PATH for any other instances of Cloud SDK tools.

    Args:
      path: str, A path to use instead of the PATH environment variable.

    Returns:
      {str}, The other executable paths that are not in the SDK root directory.
    """
        return self._FindToolsOnPath(path=path, duplicates=False, other=True)

    def FindAllDuplicateToolsOnPath(self, path=None):
        """Searches PATH for alternate versions of Cloud SDK tools in installation.

    Some gcloud components include duplicate versions of commands, so if
    those component locations are on a user's PATH then multiple versions of
    the commands may be found.

    Args:
      path: str, A path to use instead of the PATH environment variable.

    Returns:
      {str}, Alternate executable paths that are in the SDK root directory.
    """
        return self._FindToolsOnPath(path=path, duplicates=True, other=False)

    def Remove(self, ids, allow_no_backup=False):
        """Uninstalls the given components.

    Args:
      ids: list of str, The component ids to uninstall.
      allow_no_backup: bool, True if we want to allow the updater to run
        without creating a backup.  This lets us be in the root directory of the
        SDK and still do an update.  It is more fragile if there is a failure,
        so we only do it if necessary.

    Raises:
      InvalidComponentError: If any of the given component ids are not
        installed or cannot be removed.
    """
        self._CheckIfDisabledAndThrowError(components=ids, command='remove')
        if not ids:
            return
        install_state = self._GetInstallState()
        snapshot = install_state.Snapshot()
        id_set = set(ids)
        not_installed = id_set - set(snapshot.components.keys())
        if not_installed:
            raise InvalidComponentError('The following components are not currently installed [{components}]'.format(components=', '.join(not_installed)))
        to_remove = snapshot.ConsumerClosureForComponents(ids, platform_filter=self.__platform_filter)
        if self._EnableFallback():
            to_remove |= snapshot.ConsumerClosureForComponents(ids, platform_filter=self.DARWIN_X86_64)
        required_components_removed = set((c_id for c_id, component in six.iteritems(snapshot.components) if c_id in to_remove and component.is_required))
        if required_components_removed:
            raise InvalidComponentError('The following required components are included in or depend on the given components and cannot be removed [{components}]'.format(components=', '.join(required_components_removed)))
        if not to_remove:
            self.__Write(log.status, 'No components to remove.\n')
            return
        disable_backup = self._ShouldDoFastUpdate(allow_no_backup=allow_no_backup, has_components_to_remove=True)
        components_to_remove = sorted(snapshot.ComponentsFromIds(to_remove), key=lambda c: c.details.display_name)
        components_to_display = FilterMetaComponents(components_to_remove)
        self._PrintPendingAction(components_to_display, 'removed')
        self.__Write(log.status)
        config.EnsureSDKWriteAccess(self.__sdk_root)
        if self.IsPythonBundled() and _ContainsBundledPython(to_remove):
            _HaltIfBundledPythonUnix()
            log.warning(BUNDLED_PYTHON_REMOVAL_WARNING)
        self._RestartIfUsingBundledPython()
        message = self._GetDontCancelMessage(disable_backup)
        if not console_io.PromptContinue(message):
            return
        if disable_backup:
            with execution_utils.UninterruptibleSection(stream=log.status):
                self.__Write(log.status, 'Performing in place update...\n')
                self._UpdateWithProgressBar(components_to_remove, 'Uninstalling', install_state.Uninstall, first=True, last=True)
        else:
            with console_io.ProgressBar(label='Creating update staging area', stream=log.status, last=False) as pb:
                staging_state = install_state.CloneToStaging(pb.SetProgress)
            self._UpdateWithProgressBar(components_to_remove, 'Uninstalling', staging_state.Uninstall, first=False, last=False)
            with console_io.ProgressBar(label='Creating backup and activating new installation', stream=log.status, first=False) as pb:
                install_state.ReplaceWith(staging_state, pb.SetProgress)
        self._PostProcess()
        self.__Write(log.status, '\nUninstall done!\n')

    def Restore(self):
        """Restores the latest backup installation of the Cloud SDK.

    Raises:
      NoBackupError: If there is no valid backup to restore.
    """
        self._CheckIfDisabledAndThrowError()
        install_state = self._GetInstallState()
        if not install_state.HasBackup():
            raise NoBackupError('There is currently no backup to restore.')
        self._ShouldDoFastUpdate(allow_no_backup=False, fast_mode_impossible=True, has_components_to_remove=False)
        config.EnsureSDKWriteAccess(self.__sdk_root)
        backup_has_bundled_python = _ContainsBundledPython(install_state.BackupInstallationState().InstalledComponents())
        if self.IsPythonBundled() and (not backup_has_bundled_python):
            log.warning(BUNDLED_PYTHON_REMOVAL_WARNING)
        self._RestartIfUsingBundledPython()
        if not console_io.PromptContinue(message='Your Google Cloud CLI installation will be restored to its previous state.'):
            return
        self.__Write(log.status, 'Restoring backup...')
        install_state.RestoreBackup()
        self._PostProcess()
        self.__Write(log.status, 'Restoration done!\n')

    def Reinstall(self):
        """Do a reinstall of what we have based on a fresh download of the SDK.

    Returns:
      bool, True if the update succeeded, False if it was cancelled.
    """
        snapshot = self._GetLatestSnapshot(command_path='components.reinstall')
        schema_version = snapshot.sdk_definition.schema_version
        return self._DoFreshInstall(schema_version.message, schema_version.no_update, schema_version.url)

    def _ReinstallOnError(self, e):
        """Do a reinstall of what we have based on a fresh download of the SDK.

    Args:
      e: snapshots.IncompatibleSchemaVersionError, The exception we got with
        information about the new schema version.

    Returns:
      bool, True if the update succeeded, False if it was cancelled.
    """
        return self._DoFreshInstall(e.schema_version.message, e.schema_version.no_update, e.schema_version.url)

    def _DoFreshInstall(self, message, no_update, download_url):
        """Do a reinstall of what we have based on a fresh download of the SDK.

    Args:
      message: str, A message to show to the user before the re-installation.
      no_update: bool, True to show the message and tell the user they must
        re-download manually.
      download_url: The URL the Cloud SDK can be downloaded from.

    Returns:
      bool, True if the update succeeded, False if it was cancelled.
    """
        self._CheckIfDisabledAndThrowError()
        if encoding.GetEncodedValue(os.environ, 'CLOUDSDK_REINSTALL_COMPONENTS'):
            self._RaiseReinstallationFailedError()
        if message:
            self.__Write(log.status, msg=message, word_wrap=True)
        if no_update:
            return False
        config.EnsureSDKWriteAccess(self.__sdk_root)
        self._RestartIfUsingBundledPython()
        answer = console_io.PromptContinue(message='\nThe component manager must perform a self update before you can continue.  It and all components will be updated to their latest versions.')
        if not answer:
            return False
        self._ShouldDoFastUpdate(allow_no_backup=False, fast_mode_impossible=True, has_components_to_remove=False)
        install_state = self._GetInstallState()
        try:
            with console_io.ProgressBar(label='Downloading and extracting updated components', stream=log.status) as pb:
                staging_state = install_state.CreateStagingFromDownload(download_url, progress_callback=pb.SetProgress)
        except local_state.Error:
            log.error('An updated Google Cloud CLI failed to download')
            log.debug('Handling re-installation error', exc_info=True)
            self._RaiseReinstallationFailedError()
        installed_component_ids = sorted(install_state.InstalledComponents().keys())
        env = encoding.EncodeEnv(dict(os.environ))
        encoding.SetEncodedValue(env, 'CLOUDSDK_REINSTALL_COMPONENTS', ','.join(installed_component_ids))
        installer_path = os.path.join(staging_state.sdk_root, 'bin', 'bootstrapping', 'install.py')
        p = subprocess.Popen([sys.executable, '-S', installer_path], env=env)
        ret_val = p.wait()
        if ret_val:
            self._RaiseReinstallationFailedError()
        with console_io.ProgressBar(label='Creating backup and activating new installation', stream=log.status) as pb:
            install_state.ReplaceWith(staging_state, pb.SetProgress)
        self.__Write(log.status, '\nComponents updated!\n')
        return True

    def _RaiseReinstallationFailedError(self):
        raise ReinstallationFailedError('An error occurred while reinstalling the Google Cloud CLI.  Please download a new copy from: {url}'.format(url=config.INSTALLATION_CONFIG.documentation_url))

    def _EnsureInstalledAndRestart(self, components, msg=None, command=None):
        """Installs the given components if necessary and then restarts gcloud.

    Args:
      components: [str], The components that must be installed.
      msg: str, A custom message to print.
      command: str, the command to run, if not `gcloud`

    Returns:
      bool, True if the components were already installed.  If installation must
      occur, this method never returns because gcloud is reinvoked after the
      update is done.

    Raises:
      MissingRequiredComponentsError: If the components are not installed and
      the user chooses not to install them.
    """
        current_state = self._GetInstallState()
        missing_components = set(components) - set(current_state.InstalledComponents())
        if not missing_components:
            return True
        missing_components_list_str = ', '.join(missing_components)
        if not msg:
            msg = 'This action requires the installation of components: [{components}]'.format(components=missing_components_list_str)
        self.__Write(log.status, msg, word_wrap=True)
        try:
            restart_args = ['components', 'install'] + list(missing_components)
            if not self.Install(components, throw_if_unattended=True, restart_args=restart_args):
                raise MissingRequiredComponentsError("The following components are required to run this command, but are not\ncurrently installed:\n  [{components_list}]\n\nTo install them, re-run the command and choose 'yes' at the installation\nprompt, or run:\n  $ gcloud components install {components}\n\n".format(components_list=missing_components_list_str, components=' '.join(missing_components)))
        except SystemExit:
            self.__Write(log.status, 'Installing component in a new window.\n\nPlease re-run this command when installation is complete.\n    $ {0}'.format(' '.join(['gcloud'] + argv_utils.GetDecodedArgv()[1:])))
            raise
        RestartCommand(command)

    def IsPythonBundled(self):
        return _IsPythonBundled(self.__sdk_root)

    def _RestartIfUsingBundledPython(self, args=None, command=None):
        RestartIfUsingBundledPython(self.__sdk_root, args, command)

    def _GetGcloudPath(self):
        """Determines the path to the gcloud binary."""
        sdk_bin_path = config.Paths().sdk_bin_path
        if not sdk_bin_path:
            gcloud_path = file_utils.FindExecutableOnPath('gcloud')
            if gcloud_path:
                log.debug('Using gcloud found at [{path}]'.format(path=gcloud_path))
                return gcloud_path
            else:
                raise GcloudNotFoundError('A path to `gcloud` could not be found. Please check your SDK installation.')
        return os.path.join(sdk_bin_path, 'gcloud')

    def _PostProcess(self, snapshot=None):
        """Runs the gcloud command to post process the update.

    This runs gcloud as a subprocess so that the new version of gcloud (the one
    we just updated to) is run instead of the old code (which is running here).
    We do this so the new code can say how to correctly post process itself.

    Args:
      snapshot: ComponentSnapshot, The component snapshot for the version
        we are updating do. The location of gcloud and the command to run can
        change from version to version, which is why we try to pull this
        information from the latest snapshot.  For a restore operation, we don't
        have that information so we fall back to a best effort default.
    """
        command = None
        if snapshot:
            if snapshot.sdk_definition.post_processing_command:
                command = snapshot.sdk_definition.post_processing_command.split(' ')
        command = command or ['components', 'post-process']
        if self.__skip_compile_python:
            command.append('--no-compile-python')
        gcloud_path = self._GetGcloudPath()
        if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
            gcloud_args = execution_utils.ArgsForCMDTool(gcloud_path + '.cmd', *command)
        else:
            gcloud_args = execution_utils.ArgsForExecutableTool(gcloud_path, *command)
        self.__Write(log.status)
        try:
            with progress_tracker.ProgressTracker(message='Performing post processing steps', tick_delay=0.25):
                try:
                    ret_val = execution_utils.Exec(gcloud_args, no_exit=True, out_func=log.file_only_logger.debug, err_func=log.file_only_logger.debug)
                except (OSError, execution_utils.InvalidCommandError, execution_utils.PermissionError):
                    log.debug('Failed to execute post-processing command', exc_info=True)
                    raise PostProcessingError()
                if ret_val:
                    log.debug('Post-processing command exited non-zero')
                    raise PostProcessingError()
        except PostProcessingError:
            log.warning('Post processing failed.  Run `gcloud info --show-log` to view the failures.')
            self._LegacyPostProcess(snapshot)

    def _LegacyPostProcess(self, snapshot=None):
        """Runs the gcloud command to post process the update.

    This runs gcloud as a subprocess so that the new version of gcloud (the one
    we just updated to) is run instead of the old code (which is running here).
    We do this so the new code can say how to correctly post process itself.

    Args:
      snapshot: ComponentSnapshot, The component snapshot for the version
        we are updating do. The location of gcloud and the command to run can
        change from version to version, which is why we try to pull this
        information from the latest snapshot.  For a restore operation, we don't
        have that information so we fall back to a best effort default.
    """
        log.debug('Legacy post-processing...')
        command = None
        gcloud_path = None
        if snapshot:
            if snapshot.sdk_definition.post_processing_command:
                command = snapshot.sdk_definition.post_processing_command.split(' ')
            if snapshot.sdk_definition.gcloud_rel_path:
                gcloud_path = os.path.join(self.__sdk_root, snapshot.sdk_definition.gcloud_rel_path)
        command = command or ['components', 'post-process']
        if self.__skip_compile_python:
            command.append('--no-compile-python')
        gcloud_path = gcloud_path or config.GcloudPath()
        args = execution_utils.ArgsForPythonTool(gcloud_path, *command)
        try:
            with progress_tracker.ProgressTracker(message='Performing post processing steps', tick_delay=0.25):
                try:
                    ret_val = execution_utils.Exec(args, no_exit=True, out_func=log.file_only_logger.debug, err_func=log.file_only_logger.debug)
                except (OSError, execution_utils.InvalidCommandError, execution_utils.PermissionError):
                    log.debug('Failed to execute post-processing command', exc_info=True)
                    raise PostProcessingError()
                if ret_val:
                    log.debug('Post-processing command exited non-zero')
                    raise PostProcessingError()
        except PostProcessingError:
            log.warning('Post processing failed.  Run `gcloud info --show-log` to view the failures.')