from __future__ import print_function
def add_defines_to_set(header):
    symbols = []
    lineno = 0
    for line in header:
        symbol = None
        hexcode = None
        lineno += 1
        line = line.strip()
        try:
            elements = line.split()
            if line.startswith('#define'):
                symbol = elements[1]
                for element in elements:
                    if element.startswith('0x'):
                        hexcode = element
            elif line.startswith('typedef'):
                symbol = elements[-1]
            else:
                for element in elements:
                    if element.startswith('gl'):
                        symbol = element
            if symbol:
                symbols.append((symbol, lineno, line, hexcode))
        except Exception as e:
            print('error:', lineno, ':', line)
            print(e)
    return symbols