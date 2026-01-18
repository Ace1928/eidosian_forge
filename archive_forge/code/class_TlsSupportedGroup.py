import enum
import typing
class TlsSupportedGroup(enum.IntEnum):
    reserved = 0
    sect163k1 = 1
    sect163r1 = 2
    sect163r2 = 3
    sect193r1 = 4
    sect193r2 = 5
    sect233k1 = 6
    sect233r1 = 7
    sect239k1 = 8
    sect283k1 = 9
    sect283r1 = 10
    sect409k1 = 11
    sect409r1 = 12
    sect571k1 = 13
    sect571r1 = 14
    secp160k1 = 15
    secp160r1 = 16
    secp160r2 = 17
    secp192k1 = 18
    secp192r1 = 19
    secp224k1 = 20
    secp224r1 = 21
    secp256k1 = 22
    secp256r1 = 23
    secp384r1 = 24
    secp521r1 = 25
    brainpoolP256r1 = 26
    brainpoolP384r1 = 27
    brainpoolP512r1 = 28
    x25519 = 29
    x448 = 30
    brainpoolP256r1tls13 = 31
    brainpoolP384r1tls13 = 32
    brainpoolP512r1tls13 = 33
    GC256A = 34
    GC256B = 35
    GC256C = 36
    GC256D = 37
    GC512A = 38
    GC512B = 39
    GC512C = 40
    curveSM2 = 41
    ffdhe2048 = 256
    ffdhe3072 = 257
    ffdhe4096 = 258
    ffdhe6144 = 259
    ffdhe8192 = 260
    arbitrary_explicit_prime_curves = 65281
    arbitrary_explicit_char2_curves = 65282

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown Supported Group 0x{0:04X}')