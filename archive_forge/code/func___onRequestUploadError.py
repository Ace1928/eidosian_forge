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
def __onRequestUploadError(self, errorEntity, requestUploadEntity, builder, error=None):
    if error:
        return error(errorEntity.code, errorEntity.text, errorEntity.backoff)