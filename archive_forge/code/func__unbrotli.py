import zlib
from io import BytesIO
def _unbrotli(data: bytes, *, max_size: int=0) -> bytes:
    decompressor = brotli.Decompressor()
    input_stream = BytesIO(data)
    output_stream = BytesIO()
    output_chunk = b'.'
    decompressed_size = 0
    while output_chunk:
        input_chunk = input_stream.read(_CHUNK_SIZE)
        output_chunk = decompressor.process(input_chunk)
        decompressed_size += len(output_chunk)
        if max_size and decompressed_size > max_size:
            raise _DecompressionMaxSizeExceeded(f'The number of bytes decompressed so far ({decompressed_size} B) exceed the specified maximum ({max_size} B).')
        output_stream.write(output_chunk)
    output_stream.seek(0)
    return output_stream.read()