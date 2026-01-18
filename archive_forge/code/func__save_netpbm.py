from __future__ import annotations
import itertools
import math
import os
import subprocess
from enum import IntEnum
from . import (
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
def _save_netpbm(im, fp, filename):
    tempfile = im._dump()
    try:
        with open(filename, 'wb') as f:
            if im.mode != 'RGB':
                subprocess.check_call(['ppmtogif', tempfile], stdout=f, stderr=subprocess.DEVNULL)
            else:
                quant_cmd = ['ppmquant', '256', tempfile]
                togif_cmd = ['ppmtogif']
                quant_proc = subprocess.Popen(quant_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                togif_proc = subprocess.Popen(togif_cmd, stdin=quant_proc.stdout, stdout=f, stderr=subprocess.DEVNULL)
                quant_proc.stdout.close()
                retcode = quant_proc.wait()
                if retcode:
                    raise subprocess.CalledProcessError(retcode, quant_cmd)
                retcode = togif_proc.wait()
                if retcode:
                    raise subprocess.CalledProcessError(retcode, togif_cmd)
    finally:
        try:
            os.unlink(tempfile)
        except OSError:
            pass