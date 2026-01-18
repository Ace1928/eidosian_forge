from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.remote.remote_connection import RemoteConnection
class SafariRemoteConnection(RemoteConnection):
    browser_name = DesiredCapabilities.SAFARI['browserName']

    def __init__(self, remote_server_addr: str, keep_alive: bool=True, ignore_proxy: bool=False) -> None:
        super().__init__(remote_server_addr, keep_alive, ignore_proxy)
        self._commands['GET_PERMISSIONS'] = ('GET', '/session/$sessionId/apple/permissions')
        self._commands['SET_PERMISSIONS'] = ('POST', '/session/$sessionId/apple/permissions')
        self._commands['ATTACH_DEBUGGER'] = ('POST', '/session/$sessionId/apple/attach_debugger')