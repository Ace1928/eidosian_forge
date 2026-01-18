from yowsup.layers import YowParallelLayer
import time, logging, random
from yowsup.layers import YowLayer
from yowsup.layers.noise.layer import YowNoiseLayer
from yowsup.layers.noise.layer_noise_segments import YowNoiseSegmentsLayer
from yowsup.layers.auth                        import YowAuthenticationProtocolLayer
from yowsup.layers.coder                       import YowCoderLayer
from yowsup.layers.logger                      import YowLoggerLayer
from yowsup.layers.network                     import YowNetworkLayer
from yowsup.layers.protocol_messages           import YowMessagesProtocolLayer
from yowsup.layers.protocol_media              import YowMediaProtocolLayer
from yowsup.layers.protocol_acks               import YowAckProtocolLayer
from yowsup.layers.protocol_receipts           import YowReceiptProtocolLayer
from yowsup.layers.protocol_groups             import YowGroupsProtocolLayer
from yowsup.layers.protocol_presence           import YowPresenceProtocolLayer
from yowsup.layers.protocol_ib                 import YowIbProtocolLayer
from yowsup.layers.protocol_notifications      import YowNotificationsProtocolLayer
from yowsup.layers.protocol_iq                 import YowIqProtocolLayer
from yowsup.layers.protocol_contacts           import YowContactsIqProtocolLayer
from yowsup.layers.protocol_chatstate          import YowChatstateProtocolLayer
from yowsup.layers.protocol_privacy            import YowPrivacyProtocolLayer
from yowsup.layers.protocol_profiles           import YowProfilesProtocolLayer
from yowsup.layers.protocol_calls import YowCallsProtocolLayer
from yowsup.common.constants import YowConstants
from yowsup.layers.axolotl import AxolotlSendLayer, AxolotlControlLayer, AxolotlReceivelayer
from yowsup.profile.profile import YowProfile
import inspect
class YowStackBuilder(object):

    def __init__(self):
        self.layers = ()
        self._props = {}

    def setProp(self, key, value):
        self._props[key] = value
        return self

    def pushDefaultLayers(self):
        defaultLayers = YowStackBuilder.getDefaultLayers()
        self.layers += defaultLayers
        return self

    def push(self, yowLayer):
        self.layers += (yowLayer,)
        return self

    def pop(self):
        self.layers = self.layers[:-1]
        return self

    def build(self):
        return YowStack(self.layers, reversed=False, props=self._props)

    @staticmethod
    def getDefaultLayers(groups=True, media=True, privacy=True, profiles=True):
        coreLayers = YowStackBuilder.getCoreLayers()
        protocolLayers = YowStackBuilder.getProtocolLayers(groups=groups, media=media, privacy=privacy, profiles=profiles)
        allLayers = coreLayers
        allLayers += (AxolotlControlLayer,)
        allLayers += (YowParallelLayer((AxolotlSendLayer, AxolotlReceivelayer)),)
        allLayers += (YowParallelLayer(protocolLayers),)
        return allLayers

    @staticmethod
    def getDefaultStack(layer=None, axolotl=False, groups=True, media=True, privacy=True, profiles=True):
        """
        :param layer: An optional layer to put on top of default stack
        :param axolotl: E2E encryption enabled/ disabled
        :return: YowStack
        """
        allLayers = YowStackBuilder.getDefaultLayers(axolotl, groups=groups, media=media, privacy=privacy, profiles=profiles)
        if layer:
            allLayers = allLayers + (layer,)
        return YowStack(allLayers, reversed=False)

    @staticmethod
    def getCoreLayers():
        return (YowLoggerLayer, YowCoderLayer, YowNoiseLayer, YowNoiseSegmentsLayer, YowNetworkLayer)[::-1]

    @staticmethod
    def getProtocolLayers(groups=True, media=True, privacy=True, profiles=True):
        layers = YOWSUP_PROTOCOL_LAYERS_BASIC
        if groups:
            layers += (YowGroupsProtocolLayer,)
        if media:
            layers += (YowMediaProtocolLayer,)
        if privacy:
            layers += (YowPrivacyProtocolLayer,)
        if profiles:
            layers += (YowProfilesProtocolLayer,)
        return layers