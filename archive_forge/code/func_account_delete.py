from .cli import Cli, clicmd
from yowsup.layers.interface import YowInterfaceLayer, ProtocolEntityCallback
from yowsup.layers import YowLayerEvent, EventCallback
from yowsup.layers.network import YowNetworkLayer
import sys
from yowsup.common import YowConstants
import datetime
import time
import os
import logging
import threading
import base64
from yowsup.layers.protocol_groups.protocolentities      import *
from yowsup.layers.protocol_presence.protocolentities    import *
from yowsup.layers.protocol_messages.protocolentities    import *
from yowsup.layers.protocol_ib.protocolentities          import *
from yowsup.layers.protocol_iq.protocolentities          import *
from yowsup.layers.protocol_contacts.protocolentities    import *
from yowsup.layers.protocol_chatstate.protocolentities   import *
from yowsup.layers.protocol_privacy.protocolentities     import *
from yowsup.layers.protocol_media.protocolentities       import *
from yowsup.layers.protocol_media.mediauploader import MediaUploader
from yowsup.layers.protocol_profiles.protocolentities    import *
from yowsup.common.tools import Jid
from yowsup.common.optionalmodules import PILOptionalModule
from yowsup.layers.axolotl.protocolentities.iq_key_get import GetKeysIqProtocolEntity
@clicmd('Delete your account')
def account_delete(self):
    if self.assertConnected():
        if self.accountDelWarnings < self.__class__.ACCOUNT_DEL_WARNINGS:
            self.accountDelWarnings += 1
            remaining = self.__class__.ACCOUNT_DEL_WARNINGS - self.accountDelWarnings
            self.output('Repeat delete command another %s times to send the delete request' % remaining, tag='Account delete Warning !!', prompt=False)
        else:
            entity = UnregisterIqProtocolEntity()
            self.toLower(entity)