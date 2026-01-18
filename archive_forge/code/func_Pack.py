import flatbuffers
from flatbuffers.compat import import_numpy
def Pack(self, builder):
    if self.operatorCodes is not None:
        operatorCodeslist = []
        for i in range(len(self.operatorCodes)):
            operatorCodeslist.append(self.operatorCodes[i].Pack(builder))
        ModelStartOperatorCodesVector(builder, len(self.operatorCodes))
        for i in reversed(range(len(self.operatorCodes))):
            builder.PrependUOffsetTRelative(operatorCodeslist[i])
        operatorCodes = builder.EndVector()
    if self.subgraphs is not None:
        subgraphslist = []
        for i in range(len(self.subgraphs)):
            subgraphslist.append(self.subgraphs[i].Pack(builder))
        ModelStartSubgraphsVector(builder, len(self.subgraphs))
        for i in reversed(range(len(self.subgraphs))):
            builder.PrependUOffsetTRelative(subgraphslist[i])
        subgraphs = builder.EndVector()
    if self.description is not None:
        description = builder.CreateString(self.description)
    if self.buffers is not None:
        bufferslist = []
        for i in range(len(self.buffers)):
            bufferslist.append(self.buffers[i].Pack(builder))
        ModelStartBuffersVector(builder, len(self.buffers))
        for i in reversed(range(len(self.buffers))):
            builder.PrependUOffsetTRelative(bufferslist[i])
        buffers = builder.EndVector()
    if self.metadataBuffer is not None:
        if np is not None and type(self.metadataBuffer) is np.ndarray:
            metadataBuffer = builder.CreateNumpyVector(self.metadataBuffer)
        else:
            ModelStartMetadataBufferVector(builder, len(self.metadataBuffer))
            for i in reversed(range(len(self.metadataBuffer))):
                builder.PrependInt32(self.metadataBuffer[i])
            metadataBuffer = builder.EndVector()
    if self.metadata is not None:
        metadatalist = []
        for i in range(len(self.metadata)):
            metadatalist.append(self.metadata[i].Pack(builder))
        ModelStartMetadataVector(builder, len(self.metadata))
        for i in reversed(range(len(self.metadata))):
            builder.PrependUOffsetTRelative(metadatalist[i])
        metadata = builder.EndVector()
    if self.signatureDefs is not None:
        signatureDefslist = []
        for i in range(len(self.signatureDefs)):
            signatureDefslist.append(self.signatureDefs[i].Pack(builder))
        ModelStartSignatureDefsVector(builder, len(self.signatureDefs))
        for i in reversed(range(len(self.signatureDefs))):
            builder.PrependUOffsetTRelative(signatureDefslist[i])
        signatureDefs = builder.EndVector()
    ModelStart(builder)
    ModelAddVersion(builder, self.version)
    if self.operatorCodes is not None:
        ModelAddOperatorCodes(builder, operatorCodes)
    if self.subgraphs is not None:
        ModelAddSubgraphs(builder, subgraphs)
    if self.description is not None:
        ModelAddDescription(builder, description)
    if self.buffers is not None:
        ModelAddBuffers(builder, buffers)
    if self.metadataBuffer is not None:
        ModelAddMetadataBuffer(builder, metadataBuffer)
    if self.metadata is not None:
        ModelAddMetadata(builder, metadata)
    if self.signatureDefs is not None:
        ModelAddSignatureDefs(builder, signatureDefs)
    model = ModelEnd(builder)
    return model