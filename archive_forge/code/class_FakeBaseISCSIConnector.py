from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_iscsi
class FakeBaseISCSIConnector(FakeConnector, base_iscsi.BaseISCSIConnector):
    pass