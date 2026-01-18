from .base import BaseCompress
from typing import Union
class ZStdCompress(BaseCompress):

    @classmethod
    def compress(cls, data: Union[str, bytes], encoding: str='utf-8', **kwargs) -> bytes:
        """
        ZStd Compress
        """
        ensure_zstd_available()
        if isinstance(data, str):
            data = data.encode(encoding)
        return zstd.compress(data)

    @classmethod
    def decompress(cls, data: Union[str, bytes], encoding: str='utf-8', **kwargs) -> bytes:
        """
        ZStd Decompress
        """
        ensure_zstd_available()
        if isinstance(data, str):
            data = data.encode(encoding=encoding)
        return zstd.decompress(data)