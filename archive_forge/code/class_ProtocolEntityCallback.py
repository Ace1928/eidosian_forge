from yowsup.layers import YowLayer, YowLayerEvent
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from yowsup.layers.protocol_media.protocolentities.iq_requestupload import RequestUploadIqProtocolEntity
from yowsup.layers.protocol_media.mediauploader import MediaUploader
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers.auth.protocolentities import StreamErrorProtocolEntity
from yowsup.layers import EventCallback
import inspect
import logging
class ProtocolEntityCallback(object):

    def __init__(self, entityType):
        self.entityType = entityType

    def __call__(self, fn):
        fn.entity_callback = self.entityType
        return fn