import typing
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.proxy import Proxy
class BaseOptions(metaclass=ABCMeta):
    """Base class for individual browser options."""
    browser_version = _BaseOptionsDescriptor('browserVersion')
    'Gets and Sets the version of the browser.\n\n    Usage\n    -----\n    - Get\n        - `self.browser_version`\n    - Set\n        - `self.browser_version` = `value`\n\n    Parameters\n    ----------\n    `value`: `str`\n\n    Returns\n    -------\n    - Get\n        - `str`\n    - Set\n        - `None`\n    '
    platform_name = _BaseOptionsDescriptor('platformName')
    'Gets and Sets name of the platform.\n\n    Usage\n    -----\n    - Get\n        - `self.platform_name`\n    - Set\n        - `self.platform_name` = `value`\n\n    Parameters\n    ----------\n    `value`: `str`\n\n    Returns\n    -------\n    - Get\n        - `str`\n    - Set\n        - `None`\n    '
    accept_insecure_certs = _BaseOptionsDescriptor('acceptInsecureCerts')
    'Gets and Set whether the session accepts insecure certificates.\n\n    Usage\n    -----\n    - Get\n        - `self.accept_insecure_certs`\n    - Set\n        - `self.accept_insecure_certs` = `value`\n\n    Parameters\n    ----------\n    `value`: `bool`\n\n    Returns\n    -------\n    - Get\n        - `bool`\n    - Set\n        - `None`\n    '
    strict_file_interactability = _BaseOptionsDescriptor('strictFileInteractability')
    'Gets and Sets whether session is about file interactability.\n\n    Usage\n    -----\n    - Get\n        - `self.strict_file_interactability`\n    - Set\n        - `self.strict_file_interactability` = `value`\n\n    Parameters\n    ----------\n    `value`: `bool`\n\n    Returns\n    -------\n    - Get\n        - `bool`\n    - Set\n        - `None`\n    '
    set_window_rect = _BaseOptionsDescriptor('setWindowRect')
    'Gets and Sets window size and position.\n\n    Usage\n    -----\n    - Get\n        - `self.set_window_rect`\n    - Set\n        - `self.set_window_rect` = `value`\n\n    Parameters\n    ----------\n    `value`: `bool`\n\n    Returns\n    -------\n    - Get\n        - `bool`\n    - Set\n        - `None`\n    '
    page_load_strategy = _PageLoadStrategyDescriptor('pageLoadStrategy')
    ':Gets and Sets page load strategy, the default is "normal".\n\n    Usage\n    -----\n    - Get\n        - `self.page_load_strategy`\n    - Set\n        - `self.page_load_strategy` = `value`\n\n    Parameters\n    ----------\n    `value`: `str`\n\n    Returns\n    -------\n    - Get\n        - `str`\n    - Set\n        - `None`\n    '
    unhandled_prompt_behavior = _UnHandledPromptBehaviorDescriptor('unhandledPromptBehavior')
    ':Gets and Sets unhandled prompt behavior, the default is "dismiss and\n    notify".\n\n    Usage\n    -----\n    - Get\n        - `self.unhandled_prompt_behavior`\n    - Set\n        - `self.unhandled_prompt_behavior` = `value`\n\n    Parameters\n    ----------\n    `value`: `str`\n\n    Returns\n    -------\n    - Get\n        - `str`\n    - Set\n        - `None`\n    '
    timeouts = _TimeoutsDescriptor('timeouts')
    ':Gets and Sets implicit timeout, pageLoad timeout and script timeout if\n    set (in milliseconds)\n\n    Usage\n    -----\n    - Get\n        - `self.timeouts`\n    - Set\n        - `self.timeouts` = `value`\n\n    Parameters\n    ----------\n    `value`: `dict`\n\n    Returns\n    -------\n    - Get\n        - `dict`\n    - Set\n        - `None`\n    '
    proxy = _ProxyDescriptor('proxy')
    'Sets and Gets Proxy.\n\n    Usage\n    -----\n    - Get\n        - `self.proxy`\n    - Set\n        - `self.proxy` = `value`\n\n    Parameters\n    ----------\n    `value`: `Proxy`\n\n    Returns\n    -------\n    - Get\n        - `Proxy`\n    - Set\n        - `None`\n    '
    enable_downloads = _BaseOptionsDescriptor('se:downloadsEnabled')
    'Gets and Sets whether session can download files.\n\n    Usage\n    -----\n    - Get\n        - `self.enable_downloads`\n    - Set\n        - `self.enable_downloads` = `value`\n\n    Parameters\n    ----------\n    `value`: `bool`\n\n    Returns\n    -------\n    - Get\n        - `bool`\n    - Set\n        - `None`\n    '

    def __init__(self) -> None:
        super().__init__()
        self._caps = self.default_capabilities
        self._proxy = None
        self.set_capability('pageLoadStrategy', PageLoadStrategy.normal)
        self.mobile_options = None

    @property
    def capabilities(self):
        return self._caps

    def set_capability(self, name, value) -> None:
        """Sets a capability."""
        self._caps[name] = value

    def enable_mobile(self, android_package: typing.Optional[str]=None, android_activity: typing.Optional[str]=None, device_serial: typing.Optional[str]=None) -> None:
        """Enables mobile browser use for browsers that support it.

        :Args:
            android_activity: The name of the android package to start
        """
        if not android_package:
            raise AttributeError('android_package must be passed in')
        self.mobile_options = {'androidPackage': android_package}
        if android_activity:
            self.mobile_options['androidActivity'] = android_activity
        if device_serial:
            self.mobile_options['androidDeviceSerial'] = device_serial

    @abstractmethod
    def to_capabilities(self):
        """Convert options into capabilities dictionary."""

    @property
    @abstractmethod
    def default_capabilities(self):
        """Return minimal capabilities necessary as a dictionary."""