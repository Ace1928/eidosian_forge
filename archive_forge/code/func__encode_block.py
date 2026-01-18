import io
import struct
import typing
def _encode_block(self, block: bytes) -> bytes:
    block_bits = self._get_bits(block)
    lr = [block_bits[x] for x in self._ip]
    left = [lr[0:32]]
    right = [lr[32:64]]
    for i in range(16):
        computed_block = self._compute_block(right[i], self._subkeys[i])
        new_r = [int(computed_block[k] != left[i][k]) for k in range(32)]
        left.append(right[i])
        right.append(new_r)
    rl = right[16] + left[16]
    encrypted_bits = [rl[x] for x in self._final_ip]
    encrypted_bytes = b''
    for i in range(0, 64, 8):
        i_byte = int(''.join([str(x) for x in encrypted_bits[i:i + 8]]), 2)
        encrypted_bytes += struct.pack('B', i_byte)
    return encrypted_bytes