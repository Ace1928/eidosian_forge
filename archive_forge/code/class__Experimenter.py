from os_ken.ofproto.oxx_fields import (
class _Experimenter(_OxsClass):
    _class = OFPXSC_EXPERIMENTER

    def __init__(self, name, num, type_):
        super(_Experimenter, self).__init__(name, num, type_)
        self.num = (self.experimenter_id, self.oxs_type)
        self.exp_type = self.oxs_field