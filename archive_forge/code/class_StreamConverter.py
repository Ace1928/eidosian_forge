import codecs
class StreamConverter(StreamWriter, StreamReader):
    encode = codecs.ascii_decode
    decode = codecs.ascii_encode