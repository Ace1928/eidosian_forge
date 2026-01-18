from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def GetUpdateManager(group_args):
    """Construct the UpdateManager to use based on the common args for the group.

  Args:
    group_args: An argparse namespace.

  Returns:
    update_manager.UpdateManager, The UpdateManager to use for the commands.
  """
    try:
        os_override = platforms.OperatingSystem.FromId(group_args.operating_system_override)
    except platforms.InvalidEnumValue as e:
        raise exceptions.InvalidArgumentException('operating-system-override', e)
    try:
        arch_override = platforms.Architecture.FromId(group_args.architecture_override)
    except platforms.InvalidEnumValue as e:
        raise exceptions.InvalidArgumentException('architecture-override', e)
    platform = platforms.Platform.Current(os_override, arch_override)
    if not os_override and (not arch_override):
        if platform.operating_system == platforms.OperatingSystem.MACOSX and platform.architecture == platforms.Architecture.x86_64:
            if platforms.Platform.IsActuallyM1ArmArchitecture():
                platform.architecture = platforms.Architecture.arm
    root = files.ExpandHomeDir(group_args.sdk_root_override) if group_args.sdk_root_override else None
    url = files.ExpandHomeDir(group_args.snapshot_url_override) if group_args.snapshot_url_override else None
    compile_python = True
    if hasattr(group_args, 'compile_python'):
        compile_python = group_args.compile_python
    if hasattr(group_args, 'no_compile_python'):
        compile_python = group_args.no_compile_python
    return update_manager.UpdateManager(sdk_root=root, url=url, platform_filter=platform, skip_compile_python=not compile_python)