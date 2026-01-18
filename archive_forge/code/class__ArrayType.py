from __future__ import absolute_import
import math, sys
class _ArrayType(object):
    is_array = True
    subtypes = ['dtype']

    def __init__(self, dtype, ndim, is_c_contig=False, is_f_contig=False, inner_contig=False, broadcasting=None):
        self.dtype = dtype
        self.ndim = ndim
        self.is_c_contig = is_c_contig
        self.is_f_contig = is_f_contig
        self.inner_contig = inner_contig or is_c_contig or is_f_contig
        self.broadcasting = broadcasting

    def __repr__(self):
        axes = [':'] * self.ndim
        if self.is_c_contig:
            axes[-1] = '::1'
        elif self.is_f_contig:
            axes[0] = '::1'
        return '%s[%s]' % (self.dtype, ', '.join(axes))