from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
class FtdAsa5500xPlatform(AbstractFtdPlatform):
    PLATFORM_MODELS = [FtdModel.FTD_ASA5506_X, FtdModel.FTD_ASA5508_X, FtdModel.FTD_ASA5516_X]

    def __init__(self, params):
        self._ftd = Ftd5500x(hostname=params['device_hostname'], login_password=params['device_password'], sudo_password=params.get('device_sudo_password') or params['device_password'])

    def install_ftd_image(self, params):
        line = self._ftd.ssh_console(ip=params['console_ip'], port=params['console_port'], username=params['console_username'], password=params['console_password'])
        try:
            rommon_server, rommon_path = self.parse_rommon_file_location(params['rommon_file_location'])
            line.rommon_to_new_image(rommon_tftp_server=rommon_server, rommon_image=rommon_path, pkg_image=params['image_file_location'], uut_ip=params['device_ip'], uut_netmask=params['device_netmask'], uut_gateway=params['device_gateway'], dns_server=params['dns_server'], search_domains=params['search_domains'], hostname=params['device_hostname'])
        finally:
            line.disconnect()