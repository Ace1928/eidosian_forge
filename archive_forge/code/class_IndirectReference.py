from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class IndirectReference(collections.namedtuple('IndirectReferenceTuple', ['object_id', 'generation'])):

    def __str__(self):
        return f'{self.object_id} {self.generation} R'

    def __bytes__(self):
        return self.__str__().encode('us-ascii')

    def __eq__(self, other):
        return other.__class__ is self.__class__ and other.object_id == self.object_id and (other.generation == self.generation)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.object_id, self.generation))