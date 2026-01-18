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
def _sendMediaMessage(self, builder, success, error=None, progress=None):
    iq = RequestUploadIqProtocolEntity(builder.mediaType, filePath=builder.getFilepath(), encrypted=builder.isEncrypted())

    def successFn(resultEntity, requestUploadEntity):
        return self.__onRequestUploadSuccess(resultEntity, requestUploadEntity, builder, success, error, progress)

    def errorFn(errorEntity, requestUploadEntity):
        return self.__onRequestUploadError(errorEntity, requestUploadEntity, error)
    self._sendIq(iq, successFn, errorFn)