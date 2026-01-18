import typing as t
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ansible.utils.vars import merge_hash
from ._reboot import reboot_host
def _ad_should_reboot(self, result: t.Dict[str, t.Any]) -> bool:
    """Check whether a reboot is to be done

        Called after the module is run and is used to check if the reboot
        should be performed. The default check is to see if reboot_required
        was returned by the module.

        Args:
            result: The module result.

        Returns:
            bool: Whether to do a reboot or not.
        """
    return result.get('reboot_required', False)