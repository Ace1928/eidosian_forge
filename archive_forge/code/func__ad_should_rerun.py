import typing as t
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ansible.utils.vars import merge_hash
from ._reboot import reboot_host
def _ad_should_rerun(self, result: t.Dict[str, t.Any]) -> bool:
    """Check whether to rerun the module.

        Called after the reboot is completed and used to check whether the
        module should be rerun. The default is to not rerun the module.

        Args:
            result: The module result.

        Returns:
            bool: Whether to rerun the module again.
        """
    return False