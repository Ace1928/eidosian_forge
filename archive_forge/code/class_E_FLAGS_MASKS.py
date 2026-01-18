class E_FLAGS_MASKS(object):
    """Masks to be used for convenience when working with E_FLAGS

    This is a simplified approach that is also used by GNU binutils
    readelf
    """
    EFM_MIPS_ABI = 61440
    EFM_MIPS_ABI_O32 = 4096
    EFM_MIPS_ABI_O64 = 8192
    EFM_MIPS_ABI_EABI32 = 12288
    EFM_MIPS_ABI_EABI64 = 16384