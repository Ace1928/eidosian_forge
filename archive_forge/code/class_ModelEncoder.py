from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
class ModelEncoder(Encoder):

    def encode(self, model, filename, file):
        """Encode the given model to the given file.  filename
        provides a hint to the file format desired.  options are
        encoder-specific, and unknown options should be ignored or
        issue warnings.
        """
        raise NotImplementedError()