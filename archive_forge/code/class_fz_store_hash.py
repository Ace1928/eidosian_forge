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
class fz_store_hash(object):
    """
    The store can be seen as a dictionary that maps keys to
    fz_storable values. In order to allow keys of different types to
    be stored, we have a structure full of functions for each key
    'type'; this fz_store_type pointer is stored with each key, and
    tells the store how to perform certain operations (like taking/
    dropping a reference, comparing two keys, outputting details for
    debugging etc).

    The store uses a hash table internally for speed where possible.
    In order for this to work, we need a mechanism for turning a
    generic 'key' into 'a hashable string'. For this purpose the
    type structure contains a make_hash_key function pointer that
    maps from a void * to a fz_store_hash structure. If
    make_hash_key function returns 0, then the key is determined not
    to be hashable, and the value is not stored in the hash table.

    Some objects can be used both as values within the store, and as
    a component of keys within the store. We refer to these objects
    as "key storable" objects. In this case, we need to take
    additional care to ensure that we do not end up keeping an item
    within the store, purely because its value is referred to by
    another key in the store.

    An example of this are fz_images in PDF files. Each fz_image is
    placed into the	store to enable it to be easily reused. When the
    image is rendered, a pixmap is generated from the image, and the
    pixmap is placed into the store so it can be reused on
    subsequent renders. The image forms part of the key for the
    pixmap.

    When we close the pdf document (and any associated pages/display
    lists etc), we drop the images from the store. This may leave us
    in the position of the images having non-zero reference counts
    purely because they are used as part of the keys for the
    pixmaps.

    We therefore use special reference counting functions to keep
    track of these "key storable" items, and hence store the number
    of references to these items that are used in keys.

    When the number of references to an object == the number of
    references to an object from keys in the store, we know that we
    can remove all the items which have that object as part of the
    key. This is done by running a pass over the store, 'reaping'
    those items.

    Reap passes are slower than we would like as they touch every
    item in the store. We therefore provide a way to 'batch' such
    reap passes together, using fz_defer_reap_start/
    fz_defer_reap_end to bracket a region in which many may be
    triggered.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    drop = property(_mupdf.fz_store_hash_drop_get, _mupdf.fz_store_hash_drop_set)

    def __init__(self):
        _mupdf.fz_store_hash_swiginit(self, _mupdf.new_fz_store_hash())
    __swig_destroy__ = _mupdf.delete_fz_store_hash