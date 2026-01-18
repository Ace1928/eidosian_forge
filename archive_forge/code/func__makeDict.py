from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging
def _makeDict(instructionList):
    opcodeDict = {}
    mnemonicDict = {}
    for op, mnemonic, argBits, name, pops, pushes in instructionList:
        assert _mnemonicPat.match(mnemonic)
        mnemonicDict[mnemonic] = (op, argBits, name)
        if argBits:
            argoffset = op
            for i in range(1 << argBits):
                opcodeDict[op + i] = (mnemonic, argBits, argoffset, name)
        else:
            opcodeDict[op] = (mnemonic, 0, 0, name)
    return (opcodeDict, mnemonicDict)