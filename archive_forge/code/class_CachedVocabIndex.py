import os, sys, datetime, re
from rdflib import Graph
from ..utils import create_file_name
from . import VocabCachingInfo
import pickle
class CachedVocabIndex:
    """
    Class to manage the cache index. Takes care of finding the vocab directory, and manages the index
    to the individual vocab data.

    The vocab directory is set to a platform specific area, unless an environment variable
    sets it explicitly. The environment variable is "PyRdfaCacheDir"

    Every time the index is changed, the index is put back (via pickle) to the directory.

    @ivar app_data_dir: directory for the vocabulary cache directory
    @ivar index_fname: the full path of the index file on the disc
    @ivar indeces: the in-memory version of the index (a directory mapping URI-s to tuples)
    @ivar options: the error handler (option) object to send warnings to
    @type options: L{options.Options}
    @ivar report: whether details on the caching should be reported
    @type report: Boolean
    @cvar vocabs: File name used for the index in the cache directory
    @cvar preference_path: Cache directories for the three major platforms (ie, mac, windows, unix)
    @type preference_path: directory, keyed by "mac", "win", and "unix"
    @cvar architectures: Various 'architectures' as returned by the python call, and their mapping on one of the major platforms. If an architecture is missing, it is considered to be "unix"
    @type architectures: directory, mapping architectures to "mac", "win", or "unix"
    """
    vocabs = 'cache_index'
    preference_path = {'mac': 'Library/Application Support/pyRdfa-cache', 'win': 'pyRdfa-cache', 'unix': '.pyRdfa-cache'}
    architectures = {'darwin': 'mac', 'nt': 'win', 'win32': 'win', 'cygwin': 'win'}

    def __init__(self, options=None):
        """
        @param options: the error handler (option) object to send warnings to
        @type options: L{options.Options}
        """
        self.options = options
        self.report = options is not None and options.vocab_cache_report
        self.app_data_dir = self._give_preference_path()
        self.index_fname = os.path.join(self.app_data_dir, self.vocabs)
        self.indeces = {}
        if not os.path.isdir(self.app_data_dir):
            try:
                os.mkdir(self.app_data_dir)
            except Exception:
                _t, value, _traceback = sys.exc_info()
                if self.report:
                    options.add_info('Could not create the vocab cache area %s' % value, VocabCachingInfo)
                return
        else:
            if not os.access(self.app_data_dir, os.R_OK):
                if self.report:
                    options.add_info('Vocab cache directory is not readable', VocabCachingInfo)
                return
            if not os.access(self.app_data_dir, os.W_OK):
                if self.report:
                    options.add_info('Vocab cache directory is not writeable, but readable', VocabCachingInfo)
                return
        if os.path.exists(self.index_fname):
            if os.access(self.index_fname, os.R_OK):
                self.indeces = _load(self.index_fname)
            elif self.report:
                options.add_info('Vocab cache index not readable', VocabCachingInfo)
        elif os.access(self.app_data_dir, os.W_OK):
            try:
                _dump(self.indeces, self.index_fname)
            except Exception:
                _t, value, _traceback = sys.exc_info()
                if self.report:
                    options.add_info('Could not create the vocabulary index %s' % value, VocabCachingInfo)
        else:
            if self.report:
                options.add_info('Vocabulary cache directory is not writeable', VocabCachingInfo)
            self.cache_writeable = False

    def add_ref(self, uri, vocab_reference):
        """
        Add a new entry to the index, possibly removing the previous one.

        @param uri: the URI that serves as a key in the index directory
        @param vocab_reference: tuple consisting of file name, modification date, and expiration date
        """
        self.indeces[uri] = vocab_reference
        try:
            _dump(self.indeces, self.index_fname)
        except Exception:
            _t, value, _traceback = sys.exc_info()
            if self.report:
                self.options.add_info('Could not store the cache index %s' % value, VocabCachingInfo)

    def get_ref(self, uri):
        """
        Get an index entry, if available, None otherwise.
        The return value is a tuple: file name, modification date, and expiration date

        @param uri: the URI that serves as a key in the index directory
        """
        if uri in self.indeces:
            return tuple(self.indeces[uri])
        else:
            return None

    def _give_preference_path(self):
        """
        Find the vocab cache directory.
        """
        from ...pyRdfa import CACHE_DIR_VAR
        if CACHE_DIR_VAR in os.environ:
            return os.environ[CACHE_DIR_VAR]
        else:
            platform = sys.platform
            if platform in self.architectures:
                system = self.architectures[platform]
            else:
                system = 'unix'
            if system == 'win':
                app_data = os.path.expandvars('%APPDATA%')
                return os.path.join(app_data, self.preference_path[system])
            else:
                return os.path.join(os.path.expanduser('~'), self.preference_path[system])