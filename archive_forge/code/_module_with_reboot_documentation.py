import typing as t
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ansible.utils.vars import merge_hash
from ._reboot import reboot_host
Called when a reboot is done.

        Called when the reboot has been performed. The sub class can use this
        to edit the result or do additional checks as needed. The default is to
        set the reboot_required return value to False if it is in the module
        result.

        Args:
            result: The module result.
            reboot_result: The result from the reboot
        