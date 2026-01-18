from lxml import etree
from .default import DefaultDeviceHandler
from ncclient.operations.third_party.sros.rpc import MdCliRawCommand, Commit
from ncclient.xml_ import BASE_NS_1_0
def add_additional_operations(self):
    operations = {'md_cli_raw_command': MdCliRawCommand, 'commit': Commit}
    return operations