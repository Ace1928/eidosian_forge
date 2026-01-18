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
class YowStack(object):
    __stack = []
    __stackInstances = []
    __detachedQueue = Queue.Queue()

    def __init__(self, stackClassesArr=None, reversed=True, props=None):
        stackClassesArr = stackClassesArr or ()
        self.__stack = stackClassesArr[::-1] if reversed else stackClassesArr
        self.__stackInstances = []
        self._props = props or {}
        self.setProp(YowNetworkLayer.PROP_ENDPOINT, YowConstants.ENDPOINTS[random.randint(0, len(YowConstants.ENDPOINTS) - 1)])
        self._construct()

    def getLayerInterface(self, YowLayerClass):
        for inst in self.__stackInstances:
            if inst.__class__ == YowLayerClass:
                return inst.getLayerInterface()
            elif inst.__class__ == YowParallelLayer:
                res = inst.getLayerInterface(YowLayerClass)
                if res:
                    return res

    def send(self, data):
        self.__stackInstances[-1].send(data)

    def receive(self, data):
        self.__stackInstances[0].receive(data)

    def setCredentials(self, credentials):
        logger.warning('setCredentials is deprecated and any passed-in keypair is ignored, use setProfile(YowProfile) instead')
        profile_name, keypair = credentials
        self.setProfile(YowProfile(profile_name))

    def setProfile(self, profile):
        """
        :param profile: profile to use.
        :return:
        """
        logger.debug('setProfile(%s)' % profile)
        self.setProp('profile', profile if isinstance(profile, YowProfile) else YowProfile(profile))

    def addLayer(self, layerClass):
        self.__stack.push(layerClass)

    def addPostConstructLayer(self, layer):
        self.__stackInstances[-1].setLayers(layer, self.__stackInstances[-2])
        layer.setLayers(None, self.__stackInstances[-1])
        self.__stackInstances.append(layer)

    def setProp(self, key, value):
        self._props[key] = value

    def getProp(self, key, default=None):
        return self._props[key] if key in self._props else default

    def emitEvent(self, yowLayerEvent):
        if not self.__stackInstances[0].onEvent(yowLayerEvent):
            self.__stackInstances[0].emitEvent(yowLayerEvent)

    def broadcastEvent(self, yowLayerEvent):
        if not self.__stackInstances[-1].onEvent(yowLayerEvent):
            self.__stackInstances[-1].broadcastEvent(yowLayerEvent)

    def execDetached(self, fn):
        self.__class__.__detachedQueue.put(fn)

    def loop(self, *args, **kwargs):
        while True:
            try:
                callback = self.__class__.__detachedQueue.get(False)
                callback()
            except Queue.Empty:
                pass
            time.sleep(0.1)

    def _construct(self):
        logger.debug('Initializing stack')
        for s in self.__stack:
            if type(s) is tuple:
                logger.warn('Implicit declaration of parallel layers in a tuple is deprecated, pass a YowParallelLayer instead')
                inst = YowParallelLayer(s)
            elif inspect.isclass(s):
                if issubclass(s, YowLayer):
                    inst = s()
                else:
                    raise ValueError('Stack must contain only subclasses of YowLayer')
            elif issubclass(s.__class__, YowLayer):
                inst = s
            else:
                raise ValueError('Stack must contain only subclasses of YowLayer')
            logger.debug('Constructed %s' % inst)
            inst.setStack(self)
            self.__stackInstances.append(inst)
        for i in range(0, len(self.__stackInstances)):
            upperLayer = self.__stackInstances[i + 1] if i + 1 < len(self.__stackInstances) else None
            lowerLayer = self.__stackInstances[i - 1] if i > 0 else None
            self.__stackInstances[i].setLayers(upperLayer, lowerLayer)

    def getLayer(self, layerIndex):
        return self.__stackInstances[layerIndex]