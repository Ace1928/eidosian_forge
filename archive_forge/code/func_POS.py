def POS(self, x=1, y=1):
    return CSI + str(y) + ';' + str(x) + 'H'