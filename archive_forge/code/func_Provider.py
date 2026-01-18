from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
def Provider(self, provider):
    return next((p for p in self.providers if p.label == provider))