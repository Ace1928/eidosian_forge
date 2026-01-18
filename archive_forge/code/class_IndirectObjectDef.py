from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class IndirectObjectDef(IndirectReference):

    def __str__(self):
        return f'{self.object_id} {self.generation} obj'