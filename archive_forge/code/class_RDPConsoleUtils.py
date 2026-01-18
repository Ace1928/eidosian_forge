from os_win.utils import baseutils
class RDPConsoleUtils(baseutils.BaseUtilsVirt):

    def get_rdp_console_port(self):
        rdp_setting_data = self._conn.Msvm_TerminalServiceSettingData()[0]
        return rdp_setting_data.ListenerPort