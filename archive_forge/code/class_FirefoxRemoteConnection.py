from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.remote.remote_connection import RemoteConnection
class FirefoxRemoteConnection(RemoteConnection):
    browser_name = DesiredCapabilities.FIREFOX['browserName']

    def __init__(self, remote_server_addr, keep_alive=True, ignore_proxy=False) -> None:
        super().__init__(remote_server_addr, keep_alive, ignore_proxy)
        self._commands['GET_CONTEXT'] = ('GET', '/session/$sessionId/moz/context')
        self._commands['SET_CONTEXT'] = ('POST', '/session/$sessionId/moz/context')
        self._commands['INSTALL_ADDON'] = ('POST', '/session/$sessionId/moz/addon/install')
        self._commands['UNINSTALL_ADDON'] = ('POST', '/session/$sessionId/moz/addon/uninstall')
        self._commands['FULL_PAGE_SCREENSHOT'] = ('GET', '/session/$sessionId/moz/screenshot/full')