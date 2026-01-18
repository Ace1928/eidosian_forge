from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def FzBuffer_fz_subset_cff_for_gids_outparams_fn(self, num_gids, symbolic, cidfont):
    """
    Helper for out-params of class method fz_buffer::ll_fz_subset_cff_for_gids() [fz_subset_cff_for_gids()].
    """
    ret, gids = ll_fz_subset_cff_for_gids(self.m_internal, num_gids, symbolic, cidfont)
    return (FzBuffer(ll_fz_keep_buffer(ret)), gids)