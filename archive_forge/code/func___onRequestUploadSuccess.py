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
def __onRequestUploadSuccess(self, resultRequestUploadIqProtocolEntity, requestUploadEntity, builder, success, error=None, progress=None):
    if resultRequestUploadIqProtocolEntity.isDuplicate():
        return success(builder.build(resultRequestUploadIqProtocolEntity.getUrl(), resultRequestUploadIqProtocolEntity.getIp()))
    else:

        def successFn(path, jid, url):
            return self.__onMediaUploadSuccess(builder, url, resultRequestUploadIqProtocolEntity.getIp(), success)

        def errorFn(path, jid, errorText):
            return self.__onMediaUploadError(builder, errorText, error)
        mediaUploader = MediaUploader(builder.jid, self.getOwnJid(), builder.getFilepath(), resultRequestUploadIqProtocolEntity.getUrl(), resultRequestUploadIqProtocolEntity.getResumeOffset(), successFn, errorFn, progress, asynchronous=True)
        mediaUploader.start()