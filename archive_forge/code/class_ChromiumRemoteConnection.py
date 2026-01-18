from selenium.webdriver.remote.remote_connection import RemoteConnection
class ChromiumRemoteConnection(RemoteConnection):

    def __init__(self, remote_server_addr: str, vendor_prefix: str, browser_name: str, keep_alive: bool=True, ignore_proxy: bool=False) -> None:
        super().__init__(remote_server_addr, keep_alive, ignore_proxy)
        self.browser_name = browser_name
        commands = self._remote_commands(vendor_prefix)
        for key, value in commands.items():
            self._commands[key] = value

    def _remote_commands(self, vendor_prefix):
        remote_commands = {'launchApp': ('POST', '/session/$sessionId/chromium/launch_app'), 'setPermissions': ('POST', '/session/$sessionId/permissions'), 'setNetworkConditions': ('POST', '/session/$sessionId/chromium/network_conditions'), 'getNetworkConditions': ('GET', '/session/$sessionId/chromium/network_conditions'), 'deleteNetworkConditions': ('DELETE', '/session/$sessionId/chromium/network_conditions'), 'executeCdpCommand': ('POST', f'/session/$sessionId/{vendor_prefix}/cdp/execute'), 'getSinks': ('GET', f'/session/$sessionId/{vendor_prefix}/cast/get_sinks'), 'getIssueMessage': ('GET', f'/session/$sessionId/{vendor_prefix}/cast/get_issue_message'), 'setSinkToUse': ('POST', f'/session/$sessionId/{vendor_prefix}/cast/set_sink_to_use'), 'startDesktopMirroring': ('POST', f'/session/$sessionId/{vendor_prefix}/cast/start_desktop_mirroring'), 'startTabMirroring': ('POST', f'/session/$sessionId/{vendor_prefix}/cast/start_tab_mirroring'), 'stopCasting': ('POST', f'/session/$sessionId/{vendor_prefix}/cast/stop_casting')}
        return remote_commands