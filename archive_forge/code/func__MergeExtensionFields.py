from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def _MergeExtensionFields(self, x):
    for ext, val in x._extension_fields.items():
        if ext.is_repeated:
            for single_val in val:
                if ext.composite_cls is None:
                    self.AddExtension(ext, single_val)
                else:
                    self.AddExtension(ext).MergeFrom(single_val)
        elif ext.composite_cls is None:
            self.SetExtension(ext, val)
        else:
            self.MutableExtension(ext).MergeFrom(val)