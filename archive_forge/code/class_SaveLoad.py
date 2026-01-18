from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
class SaveLoad:
    """Serialize/deserialize objects from disk, by equipping them with the `save()` / `load()` methods.

    Warnings
    --------
    This uses pickle internally (among other techniques), so objects must not contain unpicklable attributes
    such as lambda functions etc.

    """

    def add_lifecycle_event(self, event_name, log_level=logging.INFO, **event):
        """
        Append an event into the `lifecycle_events` attribute of this object, and also
        optionally log the event at `log_level`.

        Events are important moments during the object's life, such as "model created",
        "model saved", "model loaded", etc.

        The `lifecycle_events` attribute is persisted across object's :meth:`~gensim.utils.SaveLoad.save`
        and :meth:`~gensim.utils.SaveLoad.load` operations. It has no impact on the use of the model,
        but is useful during debugging and support.

        Set `self.lifecycle_events = None` to disable this behaviour. Calls to `add_lifecycle_event()`
        will not record events into `self.lifecycle_events` then.

        Parameters
        ----------
        event_name : str
            Name of the event. Can be any label, e.g. "created", "stored" etc.
        event : dict
            Key-value mapping to append to `self.lifecycle_events`. Should be JSON-serializable, so keep it simple.
            Can be empty.

            This method will automatically add the following key-values to `event`, so you don't have to specify them:

            - `datetime`: the current date & time
            - `gensim`: the current Gensim version
            - `python`: the current Python version
            - `platform`: the current platform
            - `event`: the name of this event
        log_level : int
            Also log the complete event dict, at the specified log level. Set to False to not log at all.

        """
        event_dict = deepcopy(event)
        event_dict['datetime'] = datetime.now().isoformat()
        event_dict['gensim'] = gensim_version
        event_dict['python'] = sys.version
        event_dict['platform'] = platform.platform()
        event_dict['event'] = event_name
        if not hasattr(self, 'lifecycle_events'):
            logger.debug('starting a new internal lifecycle event log for %s', self.__class__.__name__)
            self.lifecycle_events = []
        if log_level:
            logger.log(log_level, '%s lifecycle event %s', self.__class__.__name__, event_dict)
        if self.lifecycle_events is not None:
            self.lifecycle_events.append(event_dict)

    @classmethod
    def load(cls, fname, mmap=None):
        """Load an object previously saved using :meth:`~gensim.utils.SaveLoad.save` from a file.

        Parameters
        ----------
        fname : str
            Path to file that contains needed object.
        mmap : str, optional
            Memory-map option.  If the object was saved with large arrays stored separately, you can load these arrays
            via mmap (shared memory) using `mmap='r'.
            If the file being loaded is compressed (either '.gz' or '.bz2'), then `mmap=None` **must be** set.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.save`
            Save object to file.

        Returns
        -------
        object
            Object loaded from `fname`.

        Raises
        ------
        AttributeError
            When called on an object instance instead of class (this is a class method).

        """
        logger.info('loading %s object from %s', cls.__name__, fname)
        compress, subname = SaveLoad._adapt_by_suffix(fname)
        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        obj.add_lifecycle_event('loaded', fname=fname)
        return obj

    def _load_specials(self, fname, mmap, compress, subname):
        """Load attributes that were stored separately, and give them the same opportunity
        to recursively load using the :class:`~gensim.utils.SaveLoad` interface.

        Parameters
        ----------
        fname : str
            Input file path.
        mmap :  {None, ‘r+’, ‘r’, ‘w+’, ‘c’}
            Memory-map options. See `numpy.load(mmap_mode)
            <https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.load.html>`_.
        compress : bool
            Is the input file compressed?
        subname : str
            Attribute name. Set automatically during recursive processing.

        """

        def mmap_error(obj, filename):
            return IOError('Cannot mmap compressed object %s in file %s. ' % (obj, filename) + 'Use `load(fname, mmap=None)` or uncompress files manually.')
        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info('loading %s recursively from %s.* with mmap=%s', attrib, cfname, mmap)
            with ignore_deprecation_warning():
                getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)
        for attrib in getattr(self, '__numpys', []):
            logger.info('loading %s from %s with mmap=%s', attrib, subname(fname, attrib), mmap)
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))
                val = np.load(subname(fname, attrib))['val']
            else:
                val = np.load(subname(fname, attrib), mmap_mode=mmap)
            with ignore_deprecation_warning():
                setattr(self, attrib, val)
        for attrib in getattr(self, '__scipys', []):
            logger.info('loading %s from %s with mmap=%s', attrib, subname(fname, attrib), mmap)
            sparse = unpickle(subname(fname, attrib))
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))
                with np.load(subname(fname, attrib, 'sparse')) as f:
                    sparse.data = f['data']
                    sparse.indptr = f['indptr']
                    sparse.indices = f['indices']
            else:
                sparse.data = np.load(subname(fname, attrib, 'data'), mmap_mode=mmap)
                sparse.indptr = np.load(subname(fname, attrib, 'indptr'), mmap_mode=mmap)
                sparse.indices = np.load(subname(fname, attrib, 'indices'), mmap_mode=mmap)
            with ignore_deprecation_warning():
                setattr(self, attrib, sparse)
        for attrib in getattr(self, '__ignoreds', []):
            logger.info('setting ignored attribute %s to None', attrib)
            with ignore_deprecation_warning():
                setattr(self, attrib, None)

    @staticmethod
    def _adapt_by_suffix(fname):
        """Get compress setting and filename for numpy file compression.

        Parameters
        ----------
        fname : str
            Input filename.

        Returns
        -------
        (bool, function)
            First argument will be True if `fname` compressed.

        """
        compress, suffix = (True, 'npz') if fname.endswith('.gz') or fname.endswith('.bz2') else (False, 'npy')
        return (compress, lambda *args: '.'.join(args + (suffix,)))

    def _smart_save(self, fname, separately=None, sep_limit=10 * 1024 ** 2, ignore=frozenset(), pickle_protocol=PICKLE_PROTOCOL):
        """Save the object to a file. Used internally by :meth:`gensim.utils.SaveLoad.save()`.

        Parameters
        ----------
        fname : str
            Path to file.
        separately : list, optional
            Iterable of attributes than need to store distinctly.
        sep_limit : int, optional
            Limit for separation.
        ignore : frozenset, optional
            Attributes that shouldn't be store.
        pickle_protocol : int, optional
            Protocol number for pickle.

        Notes
        -----
        If `separately` is None, automatically detect large numpy/scipy.sparse arrays in the object being stored,
        and store them into separate files. This avoids pickle memory errors and allows mmap'ing large arrays back
        on load efficiently.

        You can also set `separately` manually, in which case it must be a list of attribute names to be stored
        in separate files. The automatic check is not performed in this case.

        """
        compress, subname = SaveLoad._adapt_by_suffix(fname)
        restores = self._save_specials(fname, separately, sep_limit, ignore, pickle_protocol, compress, subname)
        try:
            pickle(self, fname, protocol=pickle_protocol)
        finally:
            for obj, asides in restores:
                for attrib, val in asides.items():
                    with ignore_deprecation_warning():
                        setattr(obj, attrib, val)
        logger.info('saved %s', fname)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """Save aside any attributes that need to be handled separately, including
        by recursion any attributes that are themselves :class:`~gensim.utils.SaveLoad` instances.

        Parameters
        ----------
        fname : str
            Output filename.
        separately : list or None
            List of attributes to store separately.
        sep_limit : int
            Don't store arrays smaller than this separately. In bytes.
        ignore : iterable of str
            Attributes that shouldn't be stored at all.
        pickle_protocol : int
            Protocol number for pickle.
        compress : bool
            If True - compress output with :func:`numpy.savez_compressed`.
        subname : function
            Produced by :meth:`~gensim.utils.SaveLoad._adapt_by_suffix`

        Returns
        -------
        list of (obj, {attrib: value, ...})
            Settings that the caller should use to restore each object's attributes that were set aside
            during the default :func:`~gensim.utils.pickle`.

        """
        asides = {}
        sparse_matrices = (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)
        if separately is None:
            separately = []
            for attrib, val in self.__dict__.items():
                if isinstance(val, np.ndarray) and val.size >= sep_limit:
                    separately.append(attrib)
                elif isinstance(val, sparse_matrices) and val.nnz >= sep_limit:
                    separately.append(attrib)
        with ignore_deprecation_warning():
            for attrib in separately + list(ignore):
                if hasattr(self, attrib):
                    asides[attrib] = getattr(self, attrib)
                    delattr(self, attrib)
        recursive_saveloads = []
        restores = []
        for attrib, val in self.__dict__.items():
            if hasattr(val, '_save_specials'):
                recursive_saveloads.append(attrib)
                cfname = '.'.join((fname, attrib))
                restores.extend(val._save_specials(cfname, None, sep_limit, ignore, pickle_protocol, compress, subname))
        try:
            numpys, scipys, ignoreds = ([], [], [])
            for attrib, val in asides.items():
                if isinstance(val, np.ndarray) and attrib not in ignore:
                    numpys.append(attrib)
                    logger.info("storing np array '%s' to %s", attrib, subname(fname, attrib))
                    if compress:
                        np.savez_compressed(subname(fname, attrib), val=np.ascontiguousarray(val))
                    else:
                        np.save(subname(fname, attrib), np.ascontiguousarray(val))
                elif isinstance(val, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) and attrib not in ignore:
                    scipys.append(attrib)
                    logger.info("storing scipy.sparse array '%s' under %s", attrib, subname(fname, attrib))
                    if compress:
                        np.savez_compressed(subname(fname, attrib, 'sparse'), data=val.data, indptr=val.indptr, indices=val.indices)
                    else:
                        np.save(subname(fname, attrib, 'data'), val.data)
                        np.save(subname(fname, attrib, 'indptr'), val.indptr)
                        np.save(subname(fname, attrib, 'indices'), val.indices)
                    data, indptr, indices = (val.data, val.indptr, val.indices)
                    val.data, val.indptr, val.indices = (None, None, None)
                    try:
                        pickle(val, subname(fname, attrib), protocol=pickle_protocol)
                    finally:
                        val.data, val.indptr, val.indices = (data, indptr, indices)
                else:
                    logger.info('not storing attribute %s', attrib)
                    ignoreds.append(attrib)
            self.__dict__['__numpys'] = numpys
            self.__dict__['__scipys'] = scipys
            self.__dict__['__ignoreds'] = ignoreds
            self.__dict__['__recursive_saveloads'] = recursive_saveloads
        except Exception:
            for attrib, val in asides.items():
                setattr(self, attrib, val)
            raise
        return restores + [(self, asides)]

    def save(self, fname_or_handle, separately=None, sep_limit=10 * 1024 ** 2, ignore=frozenset(), pickle_protocol=PICKLE_PROTOCOL):
        """Save the object to a file.

        Parameters
        ----------
        fname_or_handle : str or file-like
            Path to output file or already opened file-like object. If the object is a file handle,
            no special array handling will be performed, all attributes will be saved to the same file.
        separately : list of str or None, optional
            If None, automatically detect large numpy/scipy.sparse arrays in the object being stored, and store
            them into separate files. This prevent memory errors for large objects, and also allows
            `memory-mapping <https://en.wikipedia.org/wiki/Mmap>`_ the large arrays for efficient
            loading and sharing the large arrays in RAM between multiple processes.

            If list of str: store these attributes into separate files. The automated size check
            is not performed in this case.
        sep_limit : int, optional
            Don't store arrays smaller than this separately. In bytes.
        ignore : frozenset of str, optional
            Attributes that shouldn't be stored at all.
        pickle_protocol : int, optional
            Protocol number for pickle.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.load`
            Load object from file.

        """
        self.add_lifecycle_event('saving', fname_or_handle=str(fname_or_handle), separately=str(separately), sep_limit=sep_limit, ignore=ignore)
        try:
            _pickle.dump(self, fname_or_handle, protocol=pickle_protocol)
            logger.info('saved %s object', self.__class__.__name__)
        except TypeError:
            self._smart_save(fname_or_handle, separately, sep_limit, ignore, pickle_protocol=pickle_protocol)