from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
class RemoteTargetFilter(TargetFilter[TRemoteConfig]):
    """Target filter for remote Ansible Core CI managed hosts."""

    def filter_profiles(self, profiles: list[THostProfile], target: IntegrationTarget) -> list[THostProfile]:
        """Filter the list of profiles, returning only those which are not skipped for the given target."""
        profiles = super().filter_profiles(profiles, target)
        skipped_profiles = [profile for profile in profiles if any((skip in target.skips for skip in get_remote_skip_aliases(profile.config)))]
        if skipped_profiles:
            configs: list[TRemoteConfig] = [profile.config for profile in skipped_profiles]
            display.warning(f'Excluding skipped hosts from inventory: {', '.join((config.name for config in configs))}')
        profiles = [profile for profile in profiles if profile not in skipped_profiles]
        return profiles

    def filter_targets(self, targets: list[IntegrationTarget], exclude: set[str]) -> None:
        """Filter the list of targets, adding any which this host profile cannot support to the provided exclude list."""
        super().filter_targets(targets, exclude)
        if len(self.configs) > 1:
            host_skips = {host.name: get_remote_skip_aliases(host) for host in self.configs}
            skipped = [target.name for target in targets if all((any((skip in target.skips for skip in skips)) for skips in host_skips.values()))]
            if skipped:
                exclude.update(skipped)
                display.warning(f'Excluding tests which do not support {', '.join(host_skips.keys())}: {', '.join(skipped)}')
        else:
            skips = get_remote_skip_aliases(self.config)
            for skip, reason in skips.items():
                self.skip(skip, reason, targets, exclude)