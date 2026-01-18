from typing import TYPE_CHECKING
class Timeouts:

    def __init__(self, implicit_wait: float=0, page_load: float=0, script: float=0) -> None:
        """Create a new Timeouts object.

        This implements https://w3c.github.io/webdriver/#timeouts.

        :Args:
         - implicit_wait - Either an int or a float. Set how many
            seconds to wait when searching for elements before
            throwing an error.
         - page_load - Either an int or a float. Set how many seconds
            to wait for a page load to complete before throwing
            an error.
         - script - Either an int or a float. Set how many seconds to
            wait for an asynchronous script to finish execution
            before throwing an error.
        """
        self.implicit_wait = implicit_wait
        self.page_load = page_load
        self.script = script
    implicit_wait = _TimeoutsDescriptor('_implicit_wait')
    'Get or set how many seconds to wait when searching for elements.\n\n    This does not set the value on the remote end.\n\n    Usage\n    -----\n    - Get\n        - `self.implicit_wait`\n    - Set\n        - `self.implicit_wait` = `value`\n\n    Parameters\n    ----------\n    `value`: `float`\n    '
    page_load = _TimeoutsDescriptor('_page_load')
    'Get or set how many seconds to wait for the page to load.\n\n    This does not set the value on the remote end.\n\n    Usage\n    -----\n    - Get\n        - `self.page_load`\n    - Set\n        - `self.page_load` = `value`\n\n    Parameters\n    ----------\n    `value`: `float`\n    '
    script = _TimeoutsDescriptor('_script')
    'Get or set how many seconds to wait for an asynchronous script to finish\n    execution.\n\n    This does not set the value on the remote end.\n\n    Usage\n    ------\n    - Get\n        - `self.script`\n    - Set\n        - `self.script` = `value`\n\n    Parameters\n    -----------\n    `value`: `float`\n    '

    def _convert(self, timeout: float) -> int:
        if isinstance(timeout, (int, float)):
            return int(float(timeout) * 1000)
        raise TypeError('Timeouts can only be an int or a float')

    def _to_json(self) -> JSONTimeouts:
        timeouts: JSONTimeouts = {}
        if self._implicit_wait:
            timeouts['implicit'] = self._implicit_wait
        if self._page_load:
            timeouts['pageLoad'] = self._page_load
        if self._script:
            timeouts['script'] = self._script
        return timeouts