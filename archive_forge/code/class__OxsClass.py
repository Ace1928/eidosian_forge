from os_ken.ofproto.oxx_fields import (
class _OxsClass(object):

    def __init__(self, name, num, type_):
        self.name = name
        self.oxs_field = num
        self.oxs_type = num | self._class << 7
        self.num = self.oxs_type
        self.type = type_