from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
class AbstractFtdPlatform(object):
    PLATFORM_MODELS = []

    def install_ftd_image(self, params):
        raise NotImplementedError('The method should be overridden in subclass')

    @classmethod
    def supports_ftd_model(cls, model):
        return model in cls.PLATFORM_MODELS

    @staticmethod
    def parse_rommon_file_location(rommon_file_location):
        rommon_url = urlparse(rommon_file_location)
        if rommon_url.scheme != 'tftp':
            raise ValueError('The ROMMON image must be downloaded from TFTP server, other protocols are not supported.')
        return (rommon_url.netloc, rommon_url.path)