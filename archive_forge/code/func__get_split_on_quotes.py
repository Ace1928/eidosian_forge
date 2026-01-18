import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def _get_split_on_quotes(self, line):
    doublequotesplits = line.split('"')
    quoted = False
    quotesplits = []
    if len(doublequotesplits) > 1 and "'" in doublequotesplits[0]:
        singlequotesplits = doublequotesplits[0].split("'")
        doublequotesplits = doublequotesplits[1:]
        while len(singlequotesplits) % 2 == 0 and len(doublequotesplits):
            singlequotesplits[-1] += '"' + doublequotesplits[0]
            doublequotesplits = doublequotesplits[1:]
            if "'" in singlequotesplits[-1]:
                singlequotesplits = singlequotesplits[:-1] + singlequotesplits[-1].split("'")
        quotesplits += singlequotesplits
    for doublequotesplit in doublequotesplits:
        if quoted:
            quotesplits.append(doublequotesplit)
        else:
            quotesplits += doublequotesplit.split("'")
            quoted = not quoted
    return quotesplits