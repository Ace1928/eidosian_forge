import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
def __get_tx(self, stacklevel):
    tx = self.__tx
    for _ in range(stacklevel):
        tx = tx.parent
    return tx