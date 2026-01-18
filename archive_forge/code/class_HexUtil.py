import codecs
class HexUtil:

    @staticmethod
    def decodeHex(hexString):
        result = decode_hex(hexString)[0]
        return result