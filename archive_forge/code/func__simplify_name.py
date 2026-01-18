import crcmod
def _simplify_name(name):
    """
    Reduce CRC definition name to a simplified form:
        * lowercase
        * dashes removed
        * spaces removed
        * any initial "CRC" string removed
    """
    name = name.lower()
    name = name.replace('-', '')
    name = name.replace(' ', '')
    if name.startswith('crc'):
        name = name[len('crc'):]
    return name