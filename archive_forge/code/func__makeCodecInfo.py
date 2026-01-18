from collections import namedtuple
import codecs
@staticmethod
def _makeCodecInfo(name, encoding_map, decoding_map):

    class Codec(codecs.Codec):

        def encode(self, input, errors='strict', charmap_encode=codecs.charmap_encode, encoding_map=encoding_map):
            return charmap_encode(input, errors, encoding_map)

        def decode(self, input, errors='strict', charmap_decode=codecs.charmap_decode, decoding_map=decoding_map):
            return charmap_decode(input, errors, decoding_map)

    class StreamWriter(Codec, codecs.StreamWriter):
        pass

    class StreamReader(Codec, codecs.StreamReader):
        pass
    C = Codec()
    return codecs.CodecInfo(C.encode, C.decode, streamreader=StreamReader, streamwriter=StreamWriter, name=name)