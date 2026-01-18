import re
import itertools
def _dataBitIterator(self, data):
    for byte in data:
        for bit in [128, 64, 32, 16, 8, 4, 2, 1]:
            yield bool(byte & bit)