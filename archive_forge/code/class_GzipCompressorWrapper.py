import io
import zlib
from joblib.backports import LooseVersion
class GzipCompressorWrapper(CompressorWrapper):

    def __init__(self):
        CompressorWrapper.__init__(self, obj=BinaryGzipFile, prefix=_GZIP_PREFIX, extension='.gz')