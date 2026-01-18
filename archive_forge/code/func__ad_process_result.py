import typing as t
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ansible.utils.vars import merge_hash
from ._reboot import reboot_host
def _ad_process_result(self, result: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    """Called at the end of the run.

        Called at the end of the plugin run for the sub class to edit the
        result as needed. The default is for the result to be returned as is.

        Args:
            result: The module result.

        Returns:
            Dict[str, Any]: The final result to return.
        """
    return result