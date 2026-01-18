from typing import Union
def _decompress7(data: Union[bytes, bytearray, memoryview], max_order: int, mem_size: int):
    decomp = Ppmd7Decoder(max_order, mem_size)
    res = decomp.decode(data)
    return res