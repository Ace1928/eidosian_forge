from os_ken.ofproto.oxx_fields import (
from os_ken.ofproto import ofproto_common
class OldONFExperimenter(_Experimenter):
    experimenter_id = ofproto_common.ONF_EXPERIMENTER_ID

    def __init__(self, name, num, type_):
        super(OldONFExperimenter, self).__init__(name, 0, type_)
        self.num = (self.experimenter_id, num)
        self.exp_type = 2560